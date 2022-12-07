import json

import numpy as np
import torch
from tensorflow.io import gfile
from transformers import BertTokenizer, BertForPreTraining
from transformers import AutoModelForCausalLM, AutoTokenizer

import facts_pseudoperplexity.perplexity_over_time as pppl

from evaluate import load


class GdeltEvaluation:

    def __init__(self, data='20220901', device=None, batch_size=16, model='Tristan/olm-bert-base-uncased-oct-2022'):

        self.model_name = model
        self.data = data

        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                self.device = "cuda"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if 'bert' in model:
            self.model_type = 'bert'
        elif 'gpt' in model:
            self.model_type = 'gpt'

        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(model)
            model = BertForPreTraining.from_pretrained(model)
        elif self.model_type == 'gpt':
            self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=True)
            model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=True)

        self.model = model.to(device)

        # TODO: fix this lol there's no batching, need pad n batch
        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                    len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    def run(self):
        print(f'RUNNING MONTH: {self.data}')

        def datagen():
            with gfile.GFile(f'gs://hugginghelen/olm/gdelt/gdelt_data_{self.data}.jsonl', 'r') as f:
                line = f.readline()
                while line:
                    yield json.loads(line)
                    line = f.readline()

        ppls = []
        x = datagen()

        perplexity = load("perplexity", module_type="metric")
        # TODO: FIX THIS TO CHOP OFF AT MAX-SEQ-LEN OR ELSE IT BREAKS
        for i, example in enumerate(x):
            headline = example['title']
            # CHANGE THIS TO INCLUDE PPL FOR GPT

            if self.model_type == 'gpt':
                p = perplexity.compute(predictions=[headline], model_id=self.model_name)
                ppls.append(p['perplexities'])
            elif self.model_type == 'bert':
                pseudoperplexity = pppl.pseudo_perplexity(self.model, self.tokenizer, headline, self.device)
                ppls.append(pseudoperplexity)
            # print(pseudoperplexity)
            # if i == 10:
            #     break

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
