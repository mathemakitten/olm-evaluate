# Using RealToxicityPrompt's media_bias_fact_check.jsonl
import json

SNAPSHOT_DATE = '20220901'

reliable_media = []
with open('media_bias_fact_check.jsonl', 'r') as f:
    media_list = f.readlines()
    for l in media_list:
        x = json.loads(l)
        if x['fact'] == 'high':
            reliable_media.append(x['source_url_normalized'])
print("hellot")

reliable_gdelt_examples = []
with open(f'gdelt_data_{SNAPSHOT_DATE}_english.jsonl', 'r') as f:
    gdelt_tmp = f.readlines()
    for i, l in enumerate(gdelt_tmp):
        if i % 100000 == 0:
            print(f"Chewing example {i} of {len(gdelt_tmp)}")
        x = json.loads(l)
        if x['domain'] in reliable_media:
            reliable_gdelt_examples.append(x)

print(f"Number of examples left: {len(reliable_gdelt_examples)}")
print('sigh')

with open(f'gdelt_data_{SNAPSHOT_DATE}_english_reliable.jsonl', 'a') as f:
    for d in reliable_gdelt_examples:
        json.dump(d, f)
        f.write('\n')