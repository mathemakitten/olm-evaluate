from transformers import BertTokenizer, BertForPreTraining

tokenizer = BertTokenizer.from_pretrained("Tristan/olm-bert-base-uncased-oct-2022")
model = BertForPreTraining.from_pretrained("Tristan/olm-bert-base-uncased-oct-2022")

print('hello')