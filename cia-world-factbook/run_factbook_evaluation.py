import difflib as dl
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from tensorflow.io import gfile
import glob
import os
import torch.nn.functional as F
import torch
import tensorflow_hub as hub
import tensorflow as tf

# Configure which snapshot date
# TODO: turn this into a script usable w fire.Fire
dates = sorted(os.listdir('factbook'))
SNAPSHOT_DATE = '20220901'
PREV_SNAPSHOT_DATE = dates[dates.index(SNAPSHOT_DATE)-1]

page_ids = [i.split('/')[-1] for i in glob.glob('/home/helen_huggingface_co/wayback-machine-scrape/factbook/20220901/*')]

# has_any_page_changed_in_four_months = 0
# total_distinct_pages = len(set([i.split('/')[-1] for i in glob.glob('/home/helen_huggingface_co/wayback-machine-scrape/factbook/*/*')]))

device = None #"cuda"
model_name = "mathemakitten/olm-gpt2-baseline-oct-2022"
model_name = 'gpt2'
model_name = 'Tristan/olm-gpt2-oct-2022'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

# for gpt2
if tokenizer.pad_token is None: # and batch_size > 1:
    existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
    # check that the model already has at least one special token defined
    # assign one of the special tokens to also be the pad token
    tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

sentence_embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# These are calculated over the total number of changed examples, NOT documents
num_times_more_current_chosen, total = 0.0, 0.0

# countries-cambodia has changes for sure between 20220901 and 20220801
for page in page_ids[:10]:

    page = 'countries-cambodia'
    files = gfile.glob(f'gs://hugginghelen/olm/factbook/*/{page}/text.txt')
    # files = sorted([os.path.join('gs://hugginghelen/olm/factbook/', d, page, 'text.txt') for d in dates])

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
    for i, l in enumerate(current):

        if not prev:  # sometimes pages are new and have no prior snapshots
            continue

        # For this line, find the closest matching line in the previous doc. If none, skip
        curr_embedding = sentence_embedder([l])
        sentences_from_previous = [sentence for sentence in prev if len(sentence) > 2]
        # print(f"HRRE: {sentences_from_previous}")
        embeddings_for_previous = sentence_embedder(sentences_from_previous)
        likelihoods = tf.matmul(curr_embedding, embeddings_for_previous, transpose_b=True)
        # Cutoff for most relevant text: 0.5 (picked arbitrarily)
        similarity = tf.reduce_max(likelihoods[0])  # similarity measurement of most similar
        most_similar_id = sorted(range(len(likelihoods[0])), key=lambda i: likelihoods[0][i], reverse=True)[0]
        most_similar_sentence = prev[most_similar_id]

        # Assuming that the corresponding previous sentence is not exactly the same
        # TODO: this incorrectly flags cosmetic changes to webpages as real changes
        if l != most_similar_sentence and float(similarity) > 0.5 and float(similarity) < 0.94 and len(most_similar_sentence) > 2 and len(l) > 2:
            total += 1.0  # valid example only if these conditions are met

            print(f"\n\nv0: {l}\nv1: {most_similar_sentence}\nsimilarity: {similarity}")

            # Get model llhs of both of these and compare

            previous_line, current_line = most_similar_sentence, l
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
                logprobs0 += logits0[0][t][tokenized_inputs0['input_ids'][0][t+1]].detach().to(device='cpu')
            for t in range(tokenized_inputs1['input_ids'].shape[1] - 1):
                logprobs1 += logits1[0][t][tokenized_inputs1['input_ids'][0][t+1]].detach().to(device='cpu')

            logprobs = [
                logprobs0.detach().to(device='cpu') / tokenized_inputs0['input_ids'][0].shape[-1],
                        logprobs1.detach().to(device='cpu') / tokenized_inputs1['input_ids'][0].shape[-1]
                        ]
            print(f"logprobs: {logprobs}")
            # chosen_ending = torch.argmax(torch.Tensor(logprobs)).detach().to(device="cpu")

            if logprobs[1] > logprobs[0]:
                num_times_more_current_chosen += 1.0

print(f"How often did this model choose the current one? {num_times_more_current_chosen / total} ({num_times_more_current_chosen} correct, {total} total examples)")
# For this snapshot, get the diff from the previous month and dump its diff for perplexity calculation
print('hello')