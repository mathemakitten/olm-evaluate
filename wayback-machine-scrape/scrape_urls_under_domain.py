import concurrent.futures

import requests
from bs4 import BeautifulSoup
from queue import Queue
from urllib.parse import urlparse
from threading import Lock


class webCrawler:

    def __init__(self, start_link):
        self.to_visit = Queue()
        self.visited = set()
        self.start_link = start_link.rstrip('/')
        self.domain_name = urlparse(start_link).hostname
        self.visited_lock = Lock()

    def get_links(self, url):
        page = requests.get(url)
        bs = BeautifulSoup(page.content, features='lxml')
        links = list(set([link.get('href') for link in bs.findAll('a')]))
        links_to_return = []

        # relative links, Nones, mailtos http:// https://
        for l in links:
            # special conditions added for CIA World Factbook
            if l.split(".")[-1].lower() not in ['jpg',
                                                'pdf'] and 'the-world-factbook' in l and 'archives' not in l and 'locator-map' not in l:
                if l and l.startswith('/'):  # relative link
                    links_to_return.append(
                        f"{urlparse(url).scheme}://{urlparse(url).hostname.rstrip('/')}{l.rstrip('/')}")
                elif l and (l.startswith('http://') or l.startswith('https://')):
                    links_to_return.append(l.rstrip('/'))
        return list(set(links_to_return))

    def worker(self):

        while True:
            try:
                curr = self.to_visit.get(timeout=1)  # TODO handle the fact that this is a blocking call
                print(f"Visiting: {curr}")
                links_to_add = self.get_links(curr)
                for l in links_to_add:
                    if l and l not in self.visited and self.domain_name in urlparse(l).hostname:
                        self.visited_lock.acquire()
                        self.visited.add(l)
                        self.visited_lock.release()
                        self.to_visit.put(l)
            except:
                return

    def run(self, num_workers=2048):  # this is multithreaded because IO-bound
        self.to_visit.put(self.start_link)
        self.visited.add(self.start_link)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.worker) for _ in range(num_workers)]

    def save_links_to_file(self):
        with open('links_scraped.txt', 'w') as f:
            for l in self.visited:
                f.writelines(l + '\n')


if __name__ == '__main__':
    import time

    st = time.time()
    wc = webCrawler('https://www.cia.gov/the-world-factbook/')
    # wc = webCrawler('https://mathemakitten.dev')
    wc.run()
    wc.save_links_to_file()
    print(f"{time.time() - st}")
