import json

import torch.nn.functional as F
from tensorflow.io import gfile
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure which snapshot date
# TODO: turn this into a script usable w fire.Fire
SNAPSHOT_DATE = '20220901'

device = "cuda"  # "cuda"
model_name = "mathemakitten/olm-gpt2-baseline-oct-2022"
model_name = 'gpt2'
# model_name = 'Tristan/olm-gpt2-oct-2022'
# model_name = 'gpt2-medium'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
model = model.to(device)
print(f"Running eval on {model_name}")

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
    for t in range(tokenized_inputs0['input_ids'].shape[1] - 1):
        logprobs0 += logits0[0][t][tokenized_inputs0['input_ids'][0][t + 1]].detach().to(device='cpu')
    for t in range(tokenized_inputs1['input_ids'].shape[1] - 1):
        logprobs1 += logits1[0][t][tokenized_inputs1['input_ids'][0][t + 1]].detach().to(device='cpu')

    # normalize for length
    logprobs = [
        logprobs0.detach().to(device='cpu') / tokenized_inputs0['input_ids'][0].shape[-1],
        logprobs1.detach().to(device='cpu') / tokenized_inputs1['input_ids'][0].shape[-1]
    ]
    # print(f"logprobs: {logprobs}")
    if logprobs[1] > logprobs[0]:
        return 1.0
    return 0.0


with gfile.GFile(f'gs://hugginghelen/olm/factbook/diffs/factbook_diffs_{SNAPSHOT_DATE}.jsonl', 'r') as f:
    data = f.readlines()

eval_data = []

for i, line in enumerate(data):
    x = json.loads(line)
    eval_data.append(x)

import time

st = time.time()
print('Running inference')
for i, x in enumerate(eval_data):
    if i % 20000 == 0:
        print(f"Example {i} of {len(eval_data)}")
    total += 1.0  # valid example
    # print(f"\n\nv0: {current_sentence}\nv1: {most_similar_sentence}\nsimilarity: {similarity}")
    one_if_correct = compare_llhs(x)
    num_times_more_current_chosen += one_if_correct

print(
    f"How often did this model choose the current one? {num_times_more_current_chosen / total} ({num_times_more_current_chosen} correct, {total} total examples)")
print(f"Took {time.time() - st} s")
