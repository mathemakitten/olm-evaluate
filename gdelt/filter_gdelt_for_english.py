from langdetect import detect
from tensorflow.io import gfile
import json

SNAPSHOT_DATE = '20220901'

english_lines = []
with gfile.GFile(f'gs://hugginghelen/olm/gdelt/gdelt_data_{SNAPSHOT_DATE}.jsonl', 'r') as f:
    data = f.readlines()

    for i, line in enumerate(data):
        if i % 50000 == 0:
            print(f"Example {i} out of {len(data)}")
        try:
            x = json.loads(line)
            if len(x['title']) > 2:
                lang = detect(x['title'])
                if lang == 'en':
                    english_lines.append(x)
        except Exception as e:
            pass

# Write json to disk
print("Writing JSON to disk")
JSON_FILEPATH = f'gdelt_data_{SNAPSHOT_DATE}_english.jsonl'

with open(JSON_FILEPATH, 'w') as f:
    for d in english_lines:
        json.dump(d, f)
        f.write('\n')

# Push json to Hub
print("Pushing file to the HF Hub")
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj=JSON_FILEPATH,
                path_in_repo=JSON_FILEPATH,
                repo_id="olm/gdelt-news-headlines",
                repo_type="dataset")
