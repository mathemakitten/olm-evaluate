"""
For each domain in `links_scraped_cia_world_factbook.txt`, for each of <May 2022, June/July 2022, August 2022>,
get the first day of the month and find the closest snapshot according to the Wayback Machine, then call Wayback Machine
API again with the timestamp to get the actual content dump. Content dump is saved to list of dicts, which is then
converted to Pandas and then Parquet on disk.
"""

import queue
import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
import concurrent.futures
import time


# api = 'http://archive.org/wayback/available?url=example.com'

class SnapshotOverTime:

    def __init__(self):
        self.timestamps = ['20220901', '20220501', '20220601', '20220801']  # internet archive format is YYYYMMDDhhmmss
        with open('/home/helen_huggingface_co/cia-world-factbook/links_scraped_cia_world_factbook.txt', 'r') as f:
            pages = set(f.read().split('\n'))
        self.pages_queue = queue.Queue()
        [self.pages_queue.put(i) for i in pages if i != ' ' and i != "" and '#' not in i and "cover-gallery" not in i and "images" not in i and 'map' not in i]
        print(f"Number of pages in queue: {self.pages_queue.qsize()}")
        # print(pages)

    def worker(self):
        while True:
            try:
                page = self.pages_queue.get(timeout=1)
                for t in self.timestamps:
                    txt = ""
                    page_id = page.rstrip('/').split('https://www.cia.gov/the-world-factbook/')[-1].replace("/", "-")
                    print(f"Scraping time {t}, page {page}")

                    x = requests.get(f'http://archive.org/wayback/available?url={page}&timestamp={t}')
                    while x.status_code != 200:  # retry after sleeping
                        time.sleep(5)
                        x = requests.get(f'http://archive.org/wayback/available?url={page}&timestamp={t}')
                    metadata_dict = json.loads(x.text)
                    closest_timestamp = metadata_dict[
                        'timestamp']  # think we cam get rid of this since x returns closest in time
                    snapshot_url = metadata_dict['archived_snapshots']['closest']['url']

                    # scrape actual snapshot content
                    html = requests.get(snapshot_url).content
                    soup = BeautifulSoup(html, "html.parser")

                    para = soup.find_all(["p", "h2", "h3"])

                    for br in soup.find_all("br"):
                        br.replace_with("\n")

                    for p in para:
                        if p.name == "h2":
                            txt += (f"\n\nTopic: {p.get_text()}")
                        elif p.name == "h3":
                            txt += (f"\n{p.get_text()}: ")
                        else:
                            txt += (f"{p.get_text()}")

                    page_path = f"factbook/{t}/{page_id}"
                    Path(page_path).mkdir(parents=True, exist_ok=True)
                    with open(f'{page_path}/text.txt', 'w') as f:
                        f.write(txt)
            except Exception as e:
                print(e)
                print(f"Broken json for page {page_id}: {x.text}")
                return

    def run(self, num_workers=1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.worker) for _ in range(num_workers)]


s = SnapshotOverTime()
s.run()
