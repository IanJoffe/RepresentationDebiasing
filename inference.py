import argparse
import json
import submitit
import os, tempfile
from pathlib import Path
import pickle
from dataclasses import replace as dc_replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import penzai
from penzai import pz
from penzai.toolshed import save_intermediates
from penzai.models.transformer import variants

import kagglehub
import orbax.checkpoint
from jax.experimental import mesh_utils
import sentencepiece as spm

def atomic_pickle_dump(obj, target_path):
    with tempfile.NamedTemporaryFile(dir=target_path.parent, delete=False) as tf:
        pickle.dump(obj, tf, protocol=pickle.HIGHEST_PROTOCOL)
        tf.flush()
        os.fsync(tf.fileno())
        temp_name = tf.name
    os.replace(temp_name, target_path)

def load_base_model():

    weights_dir = os.path.abspath('../.cache/kagglehub/models/google/gemma/flax/2b/2')
    ckpt_path = os.path.join(weights_dir, '2b')
    vocab_path = os.path.join(weights_dir, 'tokenizer.model')
    
    vocab = spm.SentencePieceProcessor()
    vocab.Load(vocab_path)
    
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    metadata = checkpointer.metadata(ckpt_path)
    
    n_devices = jax.local_device_count()
    sharding_devices = mesh_utils.create_device_mesh((n_devices,))
    sharding = jax.sharding.PositionalSharding(sharding_devices)
    restore_args = jax.tree_util.tree_map(
        lambda m: orbax.checkpoint.ArrayRestoreArgs(
            restore_type=jax.Array,
            sharding=sharding.reshape((1,) * (len(m.shape) - 1) + (n_devices,))
        ),
        metadata,
    )
    
    flat_params = checkpointer.restore(ckpt_path, restore_args=restore_args)
    
    model = variants.gemma.gemma_from_pretrained_checkpoint(flat_params, upcast_activations_to_float32=True)

    assert REPRESENT_AT in ["post-feedforward", "pre-feedforward"]
    if REPRESENT_AT == "post-feedforward":
        model_saving_intermediates = (
            pz.select(model)
            .at_instances_of(penzai.models.transformer.model_parts.TransformerBlock)
            .apply_and_inline(
                lambda l: [l, save_intermediates.SaveIntermediate(pz.StateVariable(None))]
            )
        )
    elif REPRESENT_AT == "pre-feedforward":
        def add_z_tap(attn):
            einsum, w_o = attn.attn_value_to_output.sublayers
            tapped_seq = pz.nn.Sequential(
                [
                    einsum,                                        
                    save_intermediates.SaveIntermediate(
                        pz.StateVariable(None)                   
                    ),
                    w_o,                                          
                ]
            )
            return dc_replace(attn, attn_value_to_output=tapped_seq)

        model_saving_intermediates = (
            pz.select(model)
            .at_instances_of( penzai.nn.attention.Attention)
            .apply_and_inline(lambda attn: [add_z_tap(attn)])
        )

    return model_saving_intermediates, vocab
    

def tokenize_and_pad_vectorized(prompts, vocab, pad_value=None, padding_last=False):
    if pad_value == None:
        pad_value = vocab.pad_id()
    tokenized_prompts = prompts.apply(lambda prompt: [vocab.bos_id()] + vocab.EncodeAsIds(prompt))
    max_length = tokenized_prompts.apply(len).max()
    if padding_last:
        padded_prompts = tokenized_prompts.apply(lambda tokens: tokens + [pad_value] * (max_length - len(tokens)))
        attention_masks = tokenized_prompts.apply(lambda tokens: [1] * len(tokens) + [0] * (max_length - len(tokens)))
    else:
        padded_prompts = tokenized_prompts.apply(lambda tokens: [pad_value] * (max_length - len(tokens)) + tokens)
        attention_masks = tokenized_prompts.apply(lambda tokens: [0] * (max_length - len(tokens)) + [1] * len(tokens))
    return jnp.array(np.stack(padded_prompts.values)), jnp.array(np.stack(attention_masks.values))
    


def main():
    print(query)

    summaries_df = pd.read_csv("bill_summaries_and_sponsors.csv")
    summaries_df["true_party"] = np.select([summaries_df["D Sponsors"] >= 4*summaries_df["R Sponsors"],
                            summaries_df["D Sponsors"]*4 <= summaries_df["R Sponsors"]],
                           [-1, 1], default=0)
    
    if DATASET == "train":
        summaries_df_sample = summaries_df.sample(n=10000, replace=False, weights=(summaries_df["true_party"]!=0), random_state=137).reset_index(drop=True)
    elif DATASET == "valid":
        training_summaries_df_sample = summaries_df.sample(n=10000, replace=False, weights=(summaries_df["true_party"]!=0), random_state=137)
        summaries_df_sample = summaries_df.drop(training_summaries_df_sample.index).sample(n=10000, random_state=137).reset_index(drop=True)
        del training_summaries_df_sample

    model_saving_intermediates, vocab = load_base_model()

    DPrompt1 = lambda x: "I am an American congressperson in the Democratic Party. I was asked to vote on the following bill: \n" + x + "\n I voted"
    RPrompt1 = lambda x: "I am an American congressperson in the Republican Party. I was asked to vote on the following bill: \n" + x + "\n I voted"
    DPrompt2 = lambda x: "I am a left-leaning American congressperson. I was asked to vote on the following bill: \n" + x + "\n I voted"
    RPrompt2 = lambda x: "I am a right-leaning American congressperson. I was asked to vote on the following bill: \n" + x + "\n I voted"
    NPrompt1 = lambda x: "I am an American congressperson. I was asked to vote on the following bill: \n" + x + "\n I voted"
    NPrompt2 = lambda x: "The following bill is up for a vote: \n" + x + "\n If I were to vote on the bill, I would vote"
    all_prompts = [DPrompt1, DPrompt2, RPrompt1, RPrompt2, NPrompt1, NPrompt2]
    num_prompts = len(all_prompts)
    
    
    D1out, D2out, R1out, R2out, N1out, N2out = [], [], [], [], [], []
    D1stream, D2stream, R1stream, R2stream, N1stream, N2stream = [], [], [], [], [], []
    outputs = [D1out, D2out, R1out, R2out, N1out, N2out]
    streams  = [D1stream, D2stream, R1stream, R2stream, N1stream, N2stream]
    pickle_files = {"D1out":"/net/scratch2/ianjoffe/outputs/unpatched/" + DATASET + "/D1out.pkl",
                    "D2out":"/net/scratch2/ianjoffe/outputs/unpatched/" + DATASET + "/D2out.pkl",
                    "R1out":"/net/scratch2/ianjoffe/outputs/unpatched/" + DATASET + "/R1out.pkl",
                    "R2out":"/net/scratch2/ianjoffe/outputs/unpatched/" + DATASET + "/R2out.pkl",
                    "N1out":"/net/scratch2/ianjoffe/outputs/unpatched/" + DATASET + "/N1out.pkl",
                    "N2out":"/net/scratch2/ianjoffe/outputs/unpatched/" + DATASET + "/N2out.pkl",
                    "D1stream":"/net/scratch2/ianjoffe/activations/unpatched/" + DATASET + "/" + REPRESENT_AT + "/D1stream.pkl",
                    "D2stream":"/net/scratch2/ianjoffe/activations/unpatched/" + DATASET + "/" + REPRESENT_AT + "/D2stream.pkl",
                    "R1stream":"/net/scratch2/ianjoffe/activations/unpatched/" + DATASET + "/" + REPRESENT_AT + "/R1stream.pkl",
                    "R2stream":"/net/scratch2/ianjoffe/activations/unpatched/" + DATASET + "/" + REPRESENT_AT + "/R2stream.pkl",
                    "N1stream":"/net/scratch2/ianjoffe/activations/unpatched/" + DATASET + "/" + REPRESENT_AT + "/N1stream.pkl",
                    "N2stream":"/net/scratch2/ianjoffe/activations/unpatched/" + DATASET + "/" + REPRESENT_AT + "/N2stream.pkl"}
    
    
    for i in range(INITIAL_POSITION, ENDING_POSITION, INFERENCE_BATCH_SIZE):

        # prepare model input for batch
        selected_rows = summaries_df_sample.iloc[i:min(i+INFERENCE_BATCH_SIZE, len(summaries_df_sample))]
        prompts = pd.concat([selected_rows["summary"].apply(p) for p in all_prompts]).reset_index(drop=True)
        tokenized_prompts, attention_masks = tokenize_and_pad_vectorized(prompts, vocab, padding_last=False)
        input_batch = pz.nx.wrap(tokenized_prompts).tag("prompt", "seq")
        input_token_pos = (pz.nx.arange("seq", input_batch.named_shape["seq"]) *
                        pz.nx.ones({"prompt":input_batch.named_shape["prompt"]}) + 1) * (input_batch != 0) - 1

        # run model and extract activations
        model_outputs = model_saving_intermediates(input_batch, token_positions=input_token_pos).untag("seq")[-1]

        intermediates = pz.nx.stack([
            saver.saved.value.untag("seq")[-1]            
            for saver in (
                pz.select(model_saving_intermediates)
                .at_instances_of(save_intermediates.SaveIntermediate)
                .get_sequence()                            
            )
        ], axis_name="layer")

        # save output and activation to runtime's data structure
        for p in range(num_prompts):
            outputs[p].append(model_outputs.untag("prompt")[p*INFERENCE_BATCH_SIZE:(p+1)*INFERENCE_BATCH_SIZE].tag("bill"))
            streams[p].append(intermediates.untag("prompt")[p*INFERENCE_BATCH_SIZE:(p+1)*INFERENCE_BATCH_SIZE].tag("bill"))

        print("Completed Inference on " + str(i) + " bill")
        
        if (i % SAVE_FREQUENCY == SAVE_FREQUENCY % INFERENCE_BATCH_SIZE) or (i >= ENDING_POSITION-INFERENCE_BATCH_SIZE):
            # save activtions and outputs persistently
            
            for p in range(num_prompts):
                outputs[p][0] = pz.nx.concatenate(outputs[p], "bill")
                streams[p][0] = pz.nx.concatenate(streams[p], "bill")

            for lst in pickle_files.keys():

                #### DO NOT SAVE OUTPUTS ####
                if "activations" not in pickle_files[lst]:
                    continue
                #############################

                if Path(pickle_files[lst]).is_file():
                    with open(pickle_files[lst], "rb") as f:
                        previous_entries = pickle.load(f)
                    updated_entries = pz.nx.concatenate([previous_entries, eval(lst)[0]], "bill")
                    atomic_pickle_dump(updated_entries, Path(pickle_files[lst]))
                else:
                    with open(pickle_files[lst], "wb") as f:
                        pickle.dump(eval(lst)[0], f) 

            # reset runtime data after saving persistently
            D1out, D2out, R1out, R2out, N1out, N2out = [], [], [], [], [], []
            D1stream, D2stream, R1stream, R2stream, N1stream, N2stream = [], [], [], [], [], []
            outputs = [D1out, D2out, R1out, R2out, N1out, N2out]
            streams  = [D1stream, D2stream, R1stream, R2stream, N1stream, N2stream]
            
            print("Saved up to bill " + str(i+INFERENCE_BATCH_SIZE-1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    args = parser.parse_args()
    query_path = Path(args.query).resolve()
    with open(query_path) as f:
        query = json.load(f)

    DATASET = query.get("DATASET")
    INFERENCE_BATCH_SIZE = query.get("INFERENCE_BATCH_SIZE")
    SAVE_FREQUENCY = query.get("SAVE_FREQUENCY")
    INITIAL_POSITION = query.get("INITIAL_POSITION")
    ENDING_POSITION = query.get("ENDING_POSITION")
    REPRESENT_AT = query.get("REPRESENT_AT")

    output_directory = Path("submitit_outputs").resolve()
    executor = submitit.AutoExecutor(folder=output_directory)
    executor.update_parameters(**query.get("slurm", {}))
    executor.submit(main)
