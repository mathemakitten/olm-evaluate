import tensorflow as tf
import torch.nn.functional as F
import json
from tensorflow.io import gfile
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure which snapshot date
# TODO: turn this into a script usable w fire.Fire
SNAPSHOT_DATE = '20220901'

device = "cpu"  # "cuda"
model_name = "mathemakitten/olm-gpt2-baseline-oct-2022"
model_name = 'gpt2'
model_name = 'Tristan/olm-gpt2-oct-2022'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

# for gpt2
if tokenizer.pad_token is None:  # and batch_size > 1:
    existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
    # check that the model already has at least one special token defined
    # assign one of the special tokens to also be the pad token
    tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

# These are calculated over the total number of changed examples, NOT documents
num_times_more_current_chosen, total = 0.0, 0.0

def compare_llhs(x):
    # Get model llhs of both of these and compare
    previous_line, current_line = x['previous'], x['current']
    tokenized_inputs0 = tokenizer(previous_line, return_tensors="pt", padding=True).to(device=device)
    tokenized_inputs1 = tokenizer(current_line, return_tensors="pt", padding=True).to(device=device)
    logits0 = model(**tokenized_inputs0).logits  # .detach().to(device="cpu", dtype=torch.float32)
    logits1 = model(**tokenized_inputs1).logits  # .detach().to(device="cpu", dtype=torch.float32)
    # turn logits into logprobs
    logits0 = F.log_softmax(logits0, dim=-1)
    logits1 = F.log_softmax(logits1, dim=-1)
    # sum the logprobs
    logprobs0, logprobs1 = 0.0, 0.0
    # TODO: normalize for length here
    for t in range(tokenized_inputs0['input_ids'].shape[1] - 1):
        logprobs0 += logits0[0][t][tokenized_inputs0['input_ids'][0][t + 1]].detach().to(device='cpu')
    for t in range(tokenized_inputs1['input_ids'].shape[1] - 1):
        logprobs1 += logits1[0][t][tokenized_inputs1['input_ids'][0][t + 1]].detach().to(device='cpu')
    logprobs = [
        logprobs0.detach().to(device='cpu') / tokenized_inputs0['input_ids'][0].shape[-1],
        logprobs1.detach().to(device='cpu') / tokenized_inputs1['input_ids'][0].shape[-1]
    ]
    print(f"logprobs: {logprobs}")
    return num_times_more_current_chosen

with gfile.GFile(f'gs://hugginghelen/olm/factbook/diffs/factbook_diffs_{SNAPSHOT_DATE}_sample.jsonl', 'r') as f:
    data = f.readlines()

eval_data = []

for i, line in enumerate(data):
    x = json.loads(line)
    eval_data.append(x)

for i, x in enumerate(eval_data):
    total += 1.0  # valid example
    # print(f"\n\nv0: {current_sentence}\nv1: {most_similar_sentence}\nsimilarity: {similarity}")
    num_times_more_current_chosen = compare_llhs(x)

print(
    f"How often did this model choose the current one? {num_times_more_current_chosen / total} ({num_times_more_current_chosen} correct, {total} total examples)")





"""
# countries-cambodia has changes for sure between 20220901 and 20220801
for page in page_ids[:10]:

    page = 'countries-cambodia'
    files = gfile.glob(f'gs://hugginghelen/olm/factbook/*/{page}/text.txt')

    curr_file_index = files.index(f'gs://hugginghelen/olm/factbook/{SNAPSHOT_DATE}/{page}/text.txt')
    prev_file_index = curr_file_index - 1

    curr_file = files[curr_file_index]
    prev_file = files[prev_file_index]

    with gfile.GFile(curr_file, 'r') as f:
        current = f.readlines()
        current = [s for s in current if len(s) > 2]

    with gfile.GFile(prev_file, 'r') as f:
        prev = f.readlines()
        prev = [s for s in prev if len(s) > 2]

    # Cannot diff entire file because nonsense like `'Khmer Will Party, : -,;,;IiCcEeIiTtFfNnSsWwHhTtGgBbPpCcEeIiP'`
    # Some paragraphs only have periods or single characters changed
    # Need to diff line-by-line

    # TODO: this makes the assumption that the files have the same number of lines and the diffs are small
    # If the line has changed in the new file,
    # this *will* be messed up if drastic changes have happened in the file
    # TODO: should probably assert same number of lines — factbook doesn't change much month to month
    # TODO: in the future, find the most similar line and diff with that

    # Calculate the normalized summed logprobs for each version, see if current one is more updated
    different_lines = []

    # TODO: find most-likely-to-be-matched sentence via sentence embedding matrix for this doc to surface most likely corresponding in prev
    for i, current_sentence in enumerate(current):

        if not prev:  # sometimes pages are new and have no prior snapshots
            continue

        total += 1.0  # valid example only if these conditions are met
        print(f"\n\nv0: {current_sentence}\nv1: {most_similar_sentence}\nsimilarity: {similarity}")

        # For this line, find the closest matching line in the previous doc. If none, skip
        curr_embedding = sentence_embedder([current_sentence])
        sentences_from_previous = [sentence for sentence in prev if len(sentence) > 2]
        embeddings_for_previous = sentence_embedder(sentences_from_previous)
        likelihoods = tf.matmul(curr_embedding, embeddings_for_previous, transpose_b=True)
        similarity = tf.reduce_max(likelihoods[0])  # how similar is the most similar one?
        most_similar_id = sorted(range(len(likelihoods[0])), key=lambda i: likelihoods[0][i], reverse=True)[0]
        most_similar_sentence = prev[most_similar_id]

        # Assuming that the corresponding previous sentence is not exactly the same
        # TODO: this may incorrectly flag cosmetic changes to webpages as real changes
        if current_sentence != most_similar_sentence and float(similarity) > 0.5 and float(similarity) < 0.94 and len(
                most_similar_sentence) > 2 and len(current_sentence) > 2:
            num_times_more_current_chosen = compare_llhs()

print(
    f"How often did this model choose the current one? {num_times_more_current_chosen / total} ({num_times_more_current_chosen} correct, {total} total examples)")
"""