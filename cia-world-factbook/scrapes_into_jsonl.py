""" Turns files of the format date/pagename.txt into a dataset where the config is the scrape date, with
columns `page_id` and `text`

OR one massive jsonl where the scrape date is a column and we just append... """

import json
import os

data_clean = []

dates = os.listdir('factbook')
for date in dates:
    pages = os.listdir(os.path.join('factbook', date))
    for p in pages:
        fullpath = os.path.join('factbook', date, p, 'text.txt')
        with open(fullpath, 'r') as f:
            print(f"Processing: {fullpath}")
            text = f.read()
            jsondict = {'snapshot_date': date, 'page_id': p, 'text': text}
            data_clean.append(jsondict)

with open('data.jsonl', 'a') as f:
    for d in data_clean:
        json.dump(d, f)
        f.write('\n')

