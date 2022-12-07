import filecmp
import glob
import os

dates = os.listdir('factbook')
page_ids = [i.split('/')[-1] for i in glob.glob('/home/helen_huggingface_co/wayback-machine-scrape/factbook/20220901/*')]

has_any_page_changed_in_four_months = 0
total_distinct_pages = len(set([i.split('/')[-1] for i in glob.glob('/home/helen_huggingface_co/wayback-machine-scrape/factbook/*/*')]))

for page in page_ids:
    files = [os.path.join('factbook', d, page, 'text.txt') for d in dates]
    print(files)
    comparisons = {}
    any_file_changed = False
    for itm in range(len(files)):
        try:
            res = filecmp.cmp(files[itm], files[itm+1])
            if not res:
                any_file_changed = True
            comparisons[str(files[itm]) + ' vs ' + str(files[itm+1])] = res
        except:
            pass
        try:
            res = filecmp.cmp(files[itm], files[itm+2])
            if not res:
                any_file_changed = True
            comparisons[str(files[itm]) + ' vs ' + str(files[itm+2])] = res
        except:
            pass
        try:
            res = filecmp.cmp(files[itm], files[itm+3])
            if not res:
                any_file_changed = True
            comparisons[str(files[itm]) + ' vs ' + str(files[itm+3])] = res
        except:
            pass
    if any_file_changed:
        has_any_page_changed_in_four_months += 1

    print(f"number of files which have changed anytime over a 4-month window: {has_any_page_changed_in_four_months}")
    print(f"total distinct pages: {total_distinct_pages}")