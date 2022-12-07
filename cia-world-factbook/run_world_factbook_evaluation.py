import json
from tensorflow.io import gfile
import torch
import numpy as np
from transformers import BertTokenizer, BertForPreTraining

import facts_pseudoperplexity.perplexity_over_time as pppl
import math


class WorldFactbookEvaluation:

    def __init__(self, data='20220501', device=None, batch_size=16, model='Tristan/olm-bert-base-uncased-oct-2022'):

        if 'bert' in model:
            self.MAX_SEQ_LEN = 512
        else:  # TODO: fix this make it 'gpt' or whatever
            self.MAX_SEQ_LEN = 1024

        self.data = data

        # Get list of files for this month
        self.article_paths = gfile.glob(f'gs://hugginghelen/olm/factbook/{data}/*')

        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                self.device = "cuda"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # FIX THIS
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

    def tokenize_and_chunk(self, article_text, max_seq_len=512):
        """ Tokenize entire article and then chop up into sequence length of MAX_SEQ_LEN, return as a list """
        tensor_input = self.tokenizer.encode(article_text, return_tensors='pt')

        # Chop up into N chunks of length MAX_SEQ_LEN
        num_chunks = math.ceil(tensor_input.shape[-1] / max_seq_len)
        article_chunks_encoded = torch.tensor_split(tensor_input, num_chunks, dim=-1)
        return article_chunks_encoded

    def pseudo_perplexity(model, tokenizer, sentence, device):  # TODO: pad and batch this
        """ This version of pseudoperplexity encodes then splits into MAX_SEQ_LEN tokens then averages """

        tensor_input = tokenizer.encode(sentence, return_tensors='pt')
        repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)

        masked_input.to(device)

        with torch.no_grad():
            outputs = model(masked_input)
        logits = outputs.prediction_logits  # batch, seq len, vocab_size
        loss = torch.nn.CrossEntropyLoss()
        output = loss(torch.transpose(logits, 1, 2), labels)
        return np.exp(output)

    def run(self):
        # For each document, break it up into MAX_SEQ_LEN chunks, calculate pseudoperplexity, then average the results
        #  to get a pseudoperplexity number for a single document

        article_ppls = []

        def datagen(filepath):
            with gfile.GFile(filepath, 'r') as f:
                text = f.read()
                return text

        for article_path in self.article_paths:
            article_text = datagen(article_path)
            encoded_article_chunks = self.tokenize_and_chunk(article_text, self.MAX_SEQ_LEN)
            ppls_for_this_article = []
            for article_chunk in encoded_article_chunks:  # TODO: pad and batch this
                pseudoperplexity = pppl.pseudo_perplexity(self.model, self.tokenizer, article_chunk, self.device)
                ppls_for_this_article.append(pseudoperplexity)

            # Append the average perplexity for the entire article to the list
            article_ppls.append(np.mean(ppls_for_this_article))

        return {"pseudo_perplexities": article_ppls, "mean_pseudo_perplexity": np.mean(article_ppls)}
