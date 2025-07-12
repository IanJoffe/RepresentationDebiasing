import argparse
import json
import submitit
import os, tempfile
from pathlib import Path
import pickle
from dataclasses import replace as dc_replace
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import penzai
from penzai import pz
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
    return model, vocab
    

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
    

# YES(bill) - NO(bill)
def llm_sentiment_difference(outputs, pos_tokenids, neg_tokenids):
    pos_sums = pz.nx.nmap(jax.scipy.special.logsumexp)(outputs.untag("vocabulary")[pos_tokenids])
    neg_sums = pz.nx.nmap(jax.scipy.special.logsumexp)(outputs.untag("vocabulary")[neg_tokenids])
    return pos_sums - neg_sums

# rate on 0 to 1 scale (YES(bill) - NO(bill))
def llm_sentiment_scores(outputs, pos_tokenids, neg_tokenids):
    assert outputs.named_shape["bill"] > 1, "llm_sentiment_scores is meant to score many bills relative to each other"
    sentiment_difference = llm_sentiment_difference(outputs, pos_tokenids, neg_tokenids)
    ranks = pz.nx.nmap(jnp.argsort)(pz.nx.nmap(jnp.argsort)(sentiment_difference.untag("bill"))).tag("bill")
    scores = ranks / ranks.named_shape["bill"]
    return scores

# (YES(bill | REP) - NO(bill |REP)) - (YES(bill | DEM) - NO(bill | DEM))
def llm_partisan_scores_1(dem_outputs, rep_outputs, pos_tokenids, neg_tokenids):
    # this method first ranks how left or right a bill is, and then compares the rank difference
    rep_sentiment_scores = llm_sentiment_scores(rep_outputs, pos_tokenids, neg_tokenids)
    dem_sentiment_scores = llm_sentiment_scores(dem_outputs, pos_tokenids, neg_tokenids)
    return rep_sentiment_scores - dem_sentiment_scores

def llm_partisan_scores_2(dem_outputs, rep_outputs, pos_tokenids, neg_tokenids):
    # this method first takes the difference in how left or right a bill is, and then ranks the differences
    # otherwise same as llm_partisan_scores_1
    rep_sentiment_diff = llm_sentiment_difference(rep_outputs, pos_tokenids, neg_tokenids)
    dem_sentiment_diff = llm_sentiment_difference(dem_outputs, pos_tokenids, neg_tokenids)
    net_sentiment_diff = rep_sentiment_diff - dem_sentiment_diff
    ranks = pz.nx.nmap(jnp.argsort)(pz.nx.nmap(jnp.argsort)(net_sentiment_diff.untag("bill"))).tag("bill")
    scores = ranks / ranks.named_shape["bill"]
    return scores

# MEAN OVER ALL BILLS ((ACTIVATIONS | REP) - (ACTIVATIONS | DEM))
def mean_activation_difference(Dstream, Rstream, selected_bill_indices="all", normalize=False):
    if selected_bill_indices == "all":
        selected_bill_indices = jnp.arange(Dstream.named_shape["bill"])
    Dstream_selected = Dstream.untag("bill")[selected_bill_indices].tag("bill")
    Rstream_selected = Rstream.untag("bill")[selected_bill_indices].tag("bill")
    difference = (Rstream_selected.untag("embedding") - Dstream_selected.untag("embedding")).tag("embedding")
    mean_difference = pz.nx.nmap(jnp.mean)(difference.untag("bill"))
    if normalize:
        mean_difference = mean_difference / pz.nx.nmap(jnp.linalg.norm)(mean_difference.untag("embedding"))
    return mean_difference

# Project arr_of_embeddings into some single direction in the embedding space, collapsing that dimension into a single "score"
def directional_distance(direction, arr_of_embeddings):
    return pz.nx.nmap(jnp.dot)(direction.untag("embedding"), arr_of_embeddings.untag("embedding"))

def mean_activation_norm(stream, layer):
    return pz.nx.nmap(jnp.linalg.norm)(
        pz.nx.nmap(jnp.mean)(
            stream.untag("layer")[layer].untag("bill")
        ).untag("embedding")
    )

@pz.pytree_dataclass
class DebiasRepresentation(pz.nn.Layer):
    """Layer that adds an untrainable bias
    
    Attributes:
    direction: a unit vector of the same dimension as the embedding
    debiasing_factor: how much to move the embedding in the given direction by
    """
    direction: pz.nx.NamedArray
    debiasing_factor: jnp.float32

    def __call__(self, x: pz.nx.NamedArray, /, **unused_side_inputs) -> pz.nx.NamedArray:
        return x - (self.debiasing_factor * self.direction) # might have to make .value

def get_debiased_model(original_model, activation_num, direction=None, DStream_data=None, RStream_data=None, factor=1, normalize_factor=True):
    assert (((DStream_data is not None) and (RStream_data is not None)) or (direction is not None)) and not (((DStream_data is not None) and (RStream_data is not None)) and (direction is not None)), \
        "Debiased model must take in either streams to calculate the direction, or the direction itself"
    assert not normalize_factor if DStream_data is None else True, \
        "Cannot normalize the factor unless provided with the full unpatched stream"
    if direction is None:
        direction = mean_activation_difference(DStream_data, RStream_data, normalize=True).untag("layer")[activation_num]
    if normalize_factor:
        factor = factor * mean_activation_norm(DStream_data, activation_num)
    return pz.select(original_model).at(
        lambda model: [layer for layer in model.body.sublayers if isinstance(layer, penzai.models.transformer.model_parts.TransformerBlock)][activation_num]
    ).insert_after(DebiasRepresentation(direction, factor))

@pz.pytree_dataclass
class DebiasRepresentationHead(pz.nn.Layer):
    direction: jnp.ndarray
    factor:    float
    head_idx:  int

    def __call__(self, z, /, **unused_side_inputs):
        """
        z shape: [batch , seq , heads , 256]
        We project the selected head onto `direction` and
        add  `factor * projection * direction`  back in place.
        """
        return z.at[{ "query_heads": self.head_idx }].add(self.factor * self.direction)
    
def make_patch_attention_head(head_idx, direction, factor):
    def _patch(attn):
        subs = list(attn.attn_value_to_output.sublayers)          # any length
        new_subs = (
            subs[:-1]                                             # everything up to (but not incl.) WO
            + [DebiasRepresentationHead(direction, factor, head_idx)]     # add exactly one edit
            + subs[-1:]                                           # keep the original WO
        )
        return dc_replace(attn, attn_value_to_output=pz.nn.Sequential(new_subs))
    return _patch 

def get_head_debiased_model(original_model, num_heads=30, directions=None, DStream_data=None, RStream_data=None, factor=1, normalize_factor=True):
    assert (((DStream_data is not None) and (RStream_data is not None)) or (directions is not None)) and not (((DStream_data is not None) and (RStream_data is not None)) and (directions is not None)), \
        "Debiased model must take in either streams to calculate the direction, or the direction itself"
    assert not normalize_factor if DStream_data is None else True, \
        "Cannot normalize the factor unless provided with the full unpatched stream"
    if directions is None:
        mean_activation_diffs = pz.nx.nmap(jnp.linalg.norm)(
                                    pz.nx.nmap(jnp.mean)(
                                        (RStream_data - DStream_data).untag("projection")
                                    )
                                )
        directions = ((RStream_data - DStream_data) / mean_activation_diffs).untag("bill").mean()
    self_similarlity = abs(pz.nx.nmap(jnp.corrcoef)(
            ((RStream_data - DStream_data) / mean_activation_diffs).untag("projection"), directions.untag("projection")
        )[0][1].untag("bill").mean())

    flat = self_similarlity.untag("layer", "query_heads").ravel()
    flat = pz.nx.nmap(jnp.where)(pz.nx.nmap(jnp.isnan)(flat), -np.inf, flat)
    idx = pz.nx.nmap(jnp.argpartition)(-flat, num_heads)[:num_heads]
    layer_ix, head_ix = pz.nx.nmap(jnp.unravel_index)(idx, (self_similarlity.named_shape["layer"], self_similarlity.named_shape["query_heads"]))
    
    debiased_model = deepcopy(original_model)

    for i in range(num_heads):
        layer = layer_ix[i]
        head = head_ix[i]
        direction = directions.untag("layer")[layer].untag("query_heads")[head]
        if normalize_factor:
            factor = factor * mean_activation_diffs.untag("bill").mean().untag("layer")[layer].untag("query_heads")[head]
        debiased_model = (
            pz.select(debiased_model)
            .at(lambda m: [
                    blk for blk in m.body.sublayers 
                    if isinstance(blk, penzai.models.transformer.model_parts.TransformerBlock)
                ][layer])
            .at_instances_of(penzai.nn.attention.Attention)
            .apply(make_patch_attention_head(head, direction, factor))
        )

    return debiased_model



def main():
    print(query)

    summaries_df = pd.read_csv("bill_summaries_and_sponsors.csv")
    summaries_df["true_party"] = np.select([summaries_df["D Sponsors"] >= 4*summaries_df["R Sponsors"],
                            summaries_df["D Sponsors"]*4 <= summaries_df["R Sponsors"]],
                           [-1, 1], default=0)
    training_summaries_df_sample = summaries_df.sample(n=10000, replace=False, weights=(summaries_df["true_party"]!=0), random_state=137)
    summaries_df_sample = summaries_df.drop(training_summaries_df_sample.index).sample(n=10000, random_state=137).reset_index(drop=True)
    del training_summaries_df_sample

    model, vocab = load_base_model()

    # for gemma-1-2b these are by orders of magnitude the most common reasonable tokens to express opinions on bills
    neg_tokens = [" no", "no", " NO", "NO", " No", "No", " against", "against", " Against", "Against", " AGAINST"] # AGAINST without a space was two tokens
    pos_tokens = [" yes", "yes", " YES", "YES", " Yes", "Yes", " for", "for", " For", "For", " FOR"]
    pos_tokenids = vocab.EncodeAsIds(pos_tokens)
    neg_tokenids = vocab.EncodeAsIds(neg_tokens)

    DPrompt1 = lambda x: "I am an American congressperson in the Democratic Party. I was asked to vote on the following bill: \n" + x + "\n I voted"
    RPrompt1 = lambda x: "I am an American congressperson in the Republican Party. I was asked to vote on the following bill: \n" + x + "\n I voted"
    DPrompt2 = lambda x: "I am a left-leaning American congressperson. I was asked to vote on the following bill: \n" + x + "\n I voted"
    RPrompt2 = lambda x: "I am a right-leaning American congressperson. I was asked to vote on the following bill: \n" + x + "\n I voted"
    NPrompt1 = lambda x: "I am an American congressperson. I was asked to vote on the following bill: \n" + x + "\n I voted"
    NPrompt2 = lambda x: "The following bill is up for a vote: \n" + x + "\n If I were to vote on the bill, I would vote"
    all_prompts_patching = [DPrompt1, DPrompt2, RPrompt1, RPrompt2, NPrompt1, NPrompt2]
    num_prompts = len(all_prompts_patching)
    
    
    D1out, D2out, R1out, R2out, N1out, N2out = [], [], [], [], [], []
    outputs = [D1out, D2out, R1out, R2out, N1out, N2out]
    saving_dir_prefix = "/net/scratch2/ianjoffe/"
    if PATCHING_LOCATION == "post-feedforward":
        pickle_files = {"D1out": saving_dir_prefix + "outputs/patched/layer" + str(PATCHING_ACTIVATION_NUM) + "/" + str(STEERING_COEF).replace("0.", "_") + "/D1out.pkl",
                        "D2out": saving_dir_prefix + "outputs/patched/layer" + str(PATCHING_ACTIVATION_NUM) + "/" + str(STEERING_COEF).replace("0.", "_") + "/D2out.pkl",
                        "R1out": saving_dir_prefix + "outputs/patched/layer" + str(PATCHING_ACTIVATION_NUM) + "/" + str(STEERING_COEF).replace("0.", "_") + "/R1out.pkl",
                        "R2out": saving_dir_prefix + "outputs/patched/layer" + str(PATCHING_ACTIVATION_NUM) + "/" + str(STEERING_COEF).replace("0.", "_") + "/R2out.pkl",
                        "N1out": saving_dir_prefix + "outputs/patched/layer" + str(PATCHING_ACTIVATION_NUM) + "/" + str(STEERING_COEF).replace("0.", "_") + "/N1out.pkl",
                        "N2out": saving_dir_prefix + "outputs/patched/layer" + str(PATCHING_ACTIVATION_NUM) + "/" + str(STEERING_COEF).replace("0.", "_") + "/N2out.pkl"}
    elif PATCHING_LOCATION == "pre-feedforward":
        pickle_files = {"D1out": saving_dir_prefix + "outputs/patched/" + str(NUM_HEADS) + "heads/" + str(STEERING_COEF).replace("0.", "_") + "/D1out.pkl",
                        "D2out": saving_dir_prefix + "outputs/patched/" + str(NUM_HEADS) + "heads/" + str(STEERING_COEF).replace("0.", "_") + "/D2out.pkl",
                        "R1out": saving_dir_prefix + "outputs/patched/" + str(NUM_HEADS) + "heads/" + str(STEERING_COEF).replace("0.", "_") + "/R1out.pkl",
                        "R2out": saving_dir_prefix + "outputs/patched/" + str(NUM_HEADS) + "heads/" + str(STEERING_COEF).replace("0.", "_") + "/R2out.pkl",
                        "N1out": saving_dir_prefix + "outputs/patched/" + str(NUM_HEADS) + "heads/" + str(STEERING_COEF).replace("0.", "_") + "/N1out.pkl",
                        "N2out": saving_dir_prefix + "outputs/patched/" + str(NUM_HEADS) + "heads/" + str(STEERING_COEF).replace("0.", "_") + "/N2out.pkl"}
    
    with open(saving_dir_prefix + "/outputs/unpatched/train/D1out.pkl", "rb") as f: D1out_original = pickle.load(f)
    with open(saving_dir_prefix + "/outputs/unpatched/train/R1out.pkl", "rb") as f: R1out_original = pickle.load(f)
    with open(saving_dir_prefix + "/activations/unpatched/train/" + PATCHING_LOCATION + "/D1stream.pkl", "rb") as f: D1stream_original = pickle.load(f)
    with open(saving_dir_prefix + "/activations/unpatched/train/" + PATCHING_LOCATION + "/R1stream.pkl", "rb") as f: R1stream_original = pickle.load(f)

    
    threshold = 0.1
    partisan_scores = llm_partisan_scores_1(D1out_original, R1out_original, pos_tokenids, neg_tokenids)
    partisan_ranks = jax.device_get(partisan_scores.untag("bill").unwrap()).argsort()
    selected_idxs = np.concatenate([partisan_ranks[:int(len(partisan_ranks) * threshold)], partisan_ranks[int(len(partisan_ranks) * (1-threshold)):]])
    
    # summaries_df_minisample = training_summaries_df_sample.iloc[selected_idxs].reset_index(drop=False, names="supersample_idx")
    
    if PATCHING_LOCATION == "post-feedforward":
        debiased_model = get_debiased_model(model, PATCHING_ACTIVATION_NUM,
                                            DStream_data=D1stream_original.untag("bill")[selected_idxs].tag("bill"),
                                            RStream_data=R1stream_original.untag("bill")[selected_idxs].tag("bill"),
                                            factor=STEERING_COEF)
    elif PATCHING_LOCATION == "pre-feedforward":
        debiased_model = get_head_debiased_model(model, num_heads=NUM_HEADS,
                                                DStream_data=D1stream_original.untag("bill")[selected_idxs].tag("bill"),
                                                RStream_data=R1stream_original.untag("bill")[selected_idxs].tag("bill"),
                                                factor=STEERING_COEF)
    
    
    for i in range(INITIAL_POSITION, ENDING_POSITION, INFERENCE_BATCH_SIZE):
    
        # prepare model input for batch
        selected_rows = summaries_df_sample.iloc[i:min(i+INFERENCE_BATCH_SIZE, len(summaries_df_sample))]
        prompts = pd.concat([selected_rows["summary"].apply(p) for p in all_prompts_patching]).reset_index(drop=True)
        tokenized_prompts, attention_masks = tokenize_and_pad_vectorized(prompts, vocab, padding_last=False)    # TODO: need to further investigate the padding
        input_batch = pz.nx.wrap(tokenized_prompts).tag("prompt", "seq")
        input_token_pos = (pz.nx.arange("seq", input_batch.named_shape["seq"]) *
                           pz.nx.ones({"prompt":input_batch.named_shape["prompt"]}) + 1) * (input_batch != 0) - 1
    
        # run model
        model_outputs = debiased_model(input_batch, token_positions=input_token_pos).untag("seq")[-1]
    
        # save output and activation to runtime's data structure
        for p in range(num_prompts):
            outputs[p].append(model_outputs.untag("prompt")[p*INFERENCE_BATCH_SIZE:(p+1)*INFERENCE_BATCH_SIZE].tag("bill"))
    
        print("Completed Inference on " + str(i + INFERENCE_BATCH_SIZE - 1) + " bill")
        
        if (i % SAVE_FREQUENCY == SAVE_FREQUENCY % INFERENCE_BATCH_SIZE) or (i >= ENDING_POSITION-INFERENCE_BATCH_SIZE):
            # save activtions and outputs persistently
            
            for p in range(num_prompts):
                outputs[p][0] = pz.nx.concatenate(outputs[p], "bill")

            for lst in pickle_files.keys():
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
            outputs = [D1out, D2out, R1out, R2out, N1out, N2out]
                    
            print("Saved up to bill " + str(i+INFERENCE_BATCH_SIZE-1) + " (inclusive)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    args = parser.parse_args()
    query_path = Path(args.query).resolve()
    with open(query_path) as f:
        query = json.load(f)

    INFERENCE_BATCH_SIZE = query.get("INFERENCE_BATCH_SIZE")
    SAVE_FREQUENCY = query.get("SAVE_FREQUENCY")
    INITIAL_POSITION = query.get("INITIAL_POSITION")
    ENDING_POSITION = query.get("ENDING_POSITION")

    PATCHING_LOCATION = query.get("PATCHING_LOCATION") # pre-feedforward OR post-feedforward
    NUM_HEADS = query.get("NUM_HEADS")
    PATCHING_ACTIVATION_NUM = query.get("PATCHING_ACTIVATION_NUM")
    STEERING_COEF = query.get("STEERING_COEF")


    output_directory = Path("submitit_outputs").resolve()
    executor = submitit.AutoExecutor(folder=output_directory)
    executor.update_parameters(**query.get("slurm", {}))
    executor.submit(main)
