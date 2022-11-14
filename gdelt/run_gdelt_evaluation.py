from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
import facts_pseudoperplexity.perplexity_over_time as pppl
import json

from transformers import BertTokenizer, BertForPreTraining


class GdeltEvaluation:

    def __init__(self, data='20220501', device=None, batch_size=16, model="bert-base-uncased"):
        #
        # self.datalines = []
        # with open(f'/home/helen_huggingface_co/olm-evaluate/gdelt/gdelt_data_{data}.jsonl', 'r') as f:
        #     for line in f:
        #         self.datalines.append(json.loads(line))

        def gendata(file_name):
            with open(f'/home/helen_huggingface_co/olm-evaluate/gdelt/gdelt_data_{data}.jsonl', 'r') as f:
                line = f.readline()
                while line:
                    yield json.loads(line)
                    line = f.readline()

        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        """
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForMaskedLM.from_pretrained(model)
        """

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')

        tokenizer = BertTokenizer.from_pretrained('Tristan/olm-bert-base-uncased-oct-2022')
        model = BertForPreTraining.from_pretrained('Tristan/olm-bert-base-uncased-oct-2022')

        model = model.to(device)

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                    len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        ppls = []
        x = gendata('blah')

        for example in x:
            headline = example['title']
            pseudoperplexity = pppl.pseudo_perplexity(model, tokenizer, headline)
            print(f"text: {headline} | ppl: {pseudoperplexity}")

        # while not
        # try:
        #     example = next(x)['title']    # TODO: batch this
        #     pseudoperplexity = pppl.pseudo_perplexity(model, tokenizer, example)
        #     print(f"ppl: {pseudoperplexity}")
        # except Exception as e:
        #     print(e)

        # return {"pseudo_perplexities": ppls, "mean_pseudo_perplexity": np.mean(ppls)}

    def run(self):
        pass