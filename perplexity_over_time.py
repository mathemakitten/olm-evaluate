# from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import BertTokenizer, BertForPreTraining
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')


def pseudo_perplexity(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)

    print(f"masked input: {masked_input}\n\nlabels: {labels}\n\nmask: {mask}")

    with torch.no_grad():
        outputs = model(masked_input)
    logits = outputs.prediction_logits  # batch, seq len, vocab_size
    loss = torch.nn.CrossEntropyLoss()
    output = loss(torch.transpose(logits, 1, 2), labels)
    return np.exp(output)


# For pages over time, do perplexity
print(pseudo_perplexity(sentence='London is the capital of the United Kingdom.', model=model, tokenizer=tokenizer))  # 1.0597
print(pseudo_perplexity(sentence='London is the capital of England.', model=model, tokenizer=tokenizer))  # 1.2914
print(pseudo_perplexity(sentence='London is the capital of Great Britain.', model=model, tokenizer=tokenizer))  # 1.0969
print(pseudo_perplexity(sentence='London is the capital of South America.', model=model, tokenizer=tokenizer))  # 6.0143
print(pseudo_perplexity(sentence='London is the capital of China.', model=model, tokenizer=tokenizer))  # 11.1210

print('gello')
