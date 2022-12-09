import glob
import os
import json
import filecmp

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.io import gfile

# Configure which snapshot date
# TODO: turn this into a script usable w fire.Fire
SNAPSHOT_DATE = '20220901'
dates = sorted(os.listdir('factbook'))
PREV_SNAPSHOT_DATE = dates[dates.index(SNAPSHOT_DATE) - 1]

page_ids = [i.split('/')[-1] for i in
            gfile.glob(f'gs://hugginghelen/olm/factbook/{SNAPSHOT_DATE}/*')]

sentence_embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# because i read these pages and found that they don't change very often, so save computation
PAGE_IDS_TO_EXCLUDE = ['field-electricity-access']

page_ids = [page for page in page_ids if page not in PAGE_IDS_TO_EXCLUDE]

print(f"Running for {len(page_ids)} pages")
different_lines = set()
for i, page in enumerate(page_ids):

    # Pages to skip to save compute
    if page in ['about', 'about-gallery-of-covers', 'about-copyright-and-contributors', 'about-customer-comments', 'about-did-you-know', 'about-faqs']:
        continue

    print(f"PAGE ID: {page}")

    if i % 100 == 0:
        print(f"Page {i} of {len(page_ids)}")

    # page = 'countries-cambodia'
    files = gfile.glob(f'gs://hugginghelen/olm/factbook/*/{page}/text.txt')

    curr_file_index = files.index(f'gs://hugginghelen/olm/factbook/{SNAPSHOT_DATE}/{page}/text.txt')
    prev_file_index = curr_file_index - 1

    curr_file = files[curr_file_index]
    prev_file = files[prev_file_index]

    # Check that file is non-empty; if it is, skip
    if tf.io.gfile.stat(curr_file).length < 100:
        print(f"skipping {curr_file}")
        continue

    # check that they both exist
    if not gfile.exists(curr_file) or not gfile.exists(prev_file):
        print(f'Skipped page {curr_file} because prev doesnt exist')
        continue

    # TODO REFACTOR TO BE GFILE FRIENDLY AND FAST, UNFORTUNATELY IT'S ON THE CLOUD
    # if filecmp.cmp(prev_file, curr_file):  # save computation if they're just the same page
    #     print(f'Skipped page {curr_file} because theyre the same')
    #     continue

    with gfile.GFile(curr_file, 'r') as f:
        current = f.read()
        current = current.split('Topic: ')
        current = [s for s in current if len(s) > 50]

    with gfile.GFile(prev_file, 'r') as f:
        prev = f.read()
        prev = prev.split('Topic: ')
        prev = [s for s in prev if len(s) > 50]

    # Cannot diff entire file because nonsense like `'Khmer Will Party, : -,;,;IiCcEeIiTtFfNnSsWwHhTtGgBbPpCcEeIiP'`
    # Some paragraphs only have periods or single characters changed
    # Calculate the normalized summed logprobs for each version, see if current one is more updated

    if not prev:  # sometimes pages are new and have no prior snapshots
        continue

    if prev == current: # if they're duplicate
        # print("Skipping because no data has changed")
        continue

    if current[0][:19] == 'CIA.gov has changed' or 'Photos of ' in current:
        continue

    # Find most-likely-to-be-matched sentence via sentence embedding matrix for this doc to surface most likely corresponding in prev
    for i, current_sentence in enumerate(current):

        # print(f"CURRENT SENTENCE: {current_sentence}")

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
            # print(f"\n\nv0: {current_sentence}\nv1: {most_similar_sentence}\nsimilarity: {similarity}")
            # different_lines.append({"snapshot_date": SNAPSHOT_DATE, "previous": most_similar_sentence, "current": current_sentence})

            different_lines.add((SNAPSHOT_DATE, most_similar_sentence, current_sentence))  # need to be hashable

        # Dedupe since there's a lot of duplicated boilerplate on factbook, like the telecommunications COVID message
        # different_lines = list(set(different_lines))

        # different_lines = [dict(t) for t in {tuple(d.items()) for d in different_lines}]
        # print(different_lines)
    # print("===========================================================================")

different_lines = list(different_lines)
data = []
for d in different_lines:
    data.append({"snapshot_date": d[0], "previous": d[1], "current": d[2]})

print('Writing file')
with open(f'factbook_diffs_{SNAPSHOT_DATE}.jsonl', 'a') as f:
    for d in data:
        json.dump(d, f)
        f.write('\n')