from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from tensorflow.io.gfile import GFile as gfile
import torch
import numpy as np
import facts_pseudoperplexity.perplexity_over_time as pppl
import json

from transformers import BertTokenizer, BertForPreTraining


class GdeltEvaluation:

    def __init__(self, data='20220501', device=None, batch_size=16, model='Tristan/olm-bert-base-uncased-oct-2022'):

        self.data = data

        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = BertTokenizer.from_pretrained(model)
        model = BertForPreTraining.from_pretrained(model)
        self.model = model.to(device)

        # TODO: fix this lol there's no batching
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

        def datagen(data):
            with gfile(f'gs://hugginghelen/olm/gdelt/gdelt_data_{data}.jsonl', 'r') as f:
                line = f.readline()
                while line:
                    yield json.loads(line)
                    line = f.readline()

        ppls = []
        x = datagen(self.data)

        for i, example in enumerate(x):
            headline = example['title']
            pseudoperplexity = pppl.pseudo_perplexity(self.model, self.tokenizer, headline)
            ppls.append(pseudoperplexity)

        return {"pseudo_perplexities": ppls, "mean_pseudo_perplexity": np.mean(ppls)}
