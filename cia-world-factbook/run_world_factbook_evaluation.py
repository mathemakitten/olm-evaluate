import json
from tensorflow.io import gfile
import torch
import numpy as np
from transformers import BertTokenizer, BertForPreTraining

import facts_pseudoperplexity.perplexity_over_time as pppl


class WorldFactbookEvaluation:

    def __init__(self, data='20220501', device=None, batch_size=16, model='Tristan/olm-bert-base-uncased-oct-2022'):
        self.data = data

        # Get list of files for this month
        article_paths = gfile.glob('gs://hugginghelen/olm/factbook/20220901/*')

        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                self.device = "cuda"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

    def pseudo_perplexity(model, tokenizer, sentence, device):
        """ This version of pseudoperplexity encodes then splits into MAX_SEQ_LEN tokens then averages """

        tensor_input = tokenizer.encode(sentence, return_tensors='pt')
        repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)

        # print(f"masked input: {masked_input}\n\nlabels: {labels}\n\nmask: {mask}")

        masked_input.to(device)

        with torch.no_grad():
            outputs = model(masked_input)
        logits = outputs.prediction_logits  # batch, seq len, vocab_size
        loss = torch.nn.CrossEntropyLoss()
        output = loss(torch.transpose(logits, 1, 2), labels)
        return np.exp(output)

    def run(self):
        # For each document, break it up into MAX_SEQ_LEN chunks, calculuate pseudoperplexity, then average the results
        #  to get a pseudoperplexity number for a single document

        def datagen(filepath):
            with gfile.GFile(f'gs://hugginghelen/olm/factbook/gdelt_data_{self.data}.jsonl', 'r') as f:
                line = f.readline()
                while line:
                    yield json.loads(line)
                    line = f.readline()

        ppls = []
        x = datagen()

        for i, example in enumerate(x):
            headline = example['title']
            pseudoperplexity = pppl.pseudo_perplexity(self.model, self.tokenizer, headline, self.device)
            ppls.append(pseudoperplexity)
            # print(pseudoperplexity)
            # if i == 10:
            #     break

        return {"pseudo_perplexities": ppls, "mean_pseudo_perplexity": np.mean(ppls)}
