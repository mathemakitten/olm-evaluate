import glob
import os
import json

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.io import gfile

# Configure which snapshot date
# TODO: turn this into a script usable w fire.Fire
SNAPSHOT_DATE = '20220901'
dates = sorted(os.listdir('factbook'))
PREV_SNAPSHOT_DATE = dates[dates.index(SNAPSHOT_DATE) - 1]

page_ids = [i.split('/')[-1] for i in
            glob.glob(f'/home/helen_huggingface_co/wayback-machine-scrape/factbook/{SNAPSHOT_DATE}/*')]

sentence_embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# because i read these pages and found that they don't change very often, so save computation
PAGE_IDS_TO_EXCLUDE = ['field-electricity-access']

page_ids = [page for page in page_ids if page not in PAGE_IDS_TO_EXCLUDE]

print(f"Running for {len(page_ids)} pages")
for page in page_ids[:50]:

    print(f"\n\nRUNNING PAGE: {page}")

    # page = 'countries-cambodia'
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

    if current == prev:  # save computation if they're just the same page
        continue

    # Cannot diff entire file because nonsense like `'Khmer Will Party, : -,;,;IiCcEeIiTtFfNnSsWwHhTtGgBbPpCcEeIiP'`
    # Some paragraphs only have periods or single characters changed

    # Calculate the normalized summed logprobs for each version, see if current one is more updated
    different_lines = []

    # Find most-likely-to-be-matched sentence via sentence embedding matrix for this doc to surface most likely corresponding in prev
    for i, current_sentence in enumerate(current):

        if not prev:  # sometimes pages are new and have no prior snapshots
            continue

        # For this line, find the closest matching line in the previous doc. If none, skip
        curr_embedding = sentence_embedder([current_sentence])
        sentences_from_previous = [sentence for sentence in prev if len(sentence) > 2]
        embeddings_for_previous = sentence_embedder(sentences_from_previous)
        likelihoods = tf.matmul(curr_embedding, embeddings_for_previous, transpose_b=True)
        similarity = tf.reduce_max(likelihoods[0])  # how similar is the most similar one?
        most_similar_id = sorted(range(len(likelihoods[0])), key=lambda i: likelihoods[0][i], reverse=True)[0]
        most_similar_sentence = prev[most_similar_id]

        # print(f"\n\nv0: {current_sentence}\nv1: {most_similar_sentence}\nsimilarity: {similarity}")
        if current_sentence != most_similar_sentence and float(similarity) > 0.8 and float(similarity) < 0.94 and len(
                most_similar_sentence) > 2 and len(current_sentence) > 2 and "est." not in current_sentence:
            # if "Topic: " in current_sentence and similarity >
            print(f"\n\nv0: {current_sentence}\nv1: {most_similar_sentence}\nsimilarity: {similarity}")
            different_lines.append({"snapshot_date": SNAPSHOT_DATE, "previous": most_similar_sentence, "current": current_sentence})

        # Dedupe since there's a lot of duplicated boilerplate on factbook, like the telecommunications COVID message
        # different_lines = list(set(different_lines))

        different_lines = [dict(t) for t in {tuple(d.items()) for d in different_lines}]
        # print(different_lines)
    print("===========================================================================")

with open(f'factbook_diffs_{SNAPSHOT_DATE}.jsonl', 'a') as f:
    for d in different_lines:
        json.dump(d, f)
        f.write('\n')