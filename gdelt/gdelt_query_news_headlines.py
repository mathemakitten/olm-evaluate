import os
import json
import csv
from google.cloud import bigquery
from google.oauth2 import service_account


credentials = service_account.Credentials.from_service_account_file('huggingface-science-eval-olm-service-acct.json')

with open('./dates.csv', 'r') as f:
    csvreader = csv.reader(f)
    header = next(csvreader)
    cc_dates = []
    for row in csvreader:
        cc_dates.append(row)

client = bigquery.Client(credentials=credentials)

previous_cc_end_date = cc_dates[0][-2]

def reformat_date_for_sql(date):
    return str(date[:4] + '-' + date[4:6] + '-' + date[6:])

prev_scrape_end_date = '2022-01-29'
for d in cc_dates[1:]:
    lines_for_jsonl = []

    cc_scrape_start_date, cc_scrape_end_date = reformat_date_for_sql(d[1]), reformat_date_for_sql(d[2])

    query_num_rows_to_process = f"SELECT count(*) FROM `gdelt-bq.gdeltv2.gal` WHERE DATE(date) > '{prev_scrape_end_date}' and DATE(date) <= '{cc_scrape_start_date}' and lang = 'en'"
    num_rows_query = client.query(query_num_rows_to_process)
    for r in num_rows_query:
        num_rows = r[0]

    print(f"date range: {prev_scrape_end_date} to {cc_scrape_start_date}")
    # query past 30 days worth of headlines prior to the 1st of the month of the new scrape
    query = f"SELECT date, url, domain, outletName, title FROM `gdelt-bq.gdeltv2.gal` WHERE DATE(date) > '{prev_scrape_end_date}' and DATE(date) <= '{cc_scrape_start_date}' and lang = 'en'"

    query_job = client.query(query)

    for i, row in enumerate(query_job):
        if i % 500000 == 0:
            print(f"Row {i} of {num_rows}")
        jsondict = {'snapshot_date': d[-1], 'article_date': row[0].strftime("%Y%m%d"), 'domain': row[2], 'url': row[1], 'outlet_name': row[3], 'title': row[4]}
        # Row values can be accessed by field name or index.
        lines_for_jsonl.append(jsondict)

    prev_scrape_end_date = cc_scrape_end_date

    # Fixed malformed data due to quotes in the headline; replace double quotes w single quotes
    with open(f'gdelt/gdelt_data_{d[-1]}.jsonl', 'a') as f:
        for d in lines_for_jsonl:
            json.dump(d, f)
            f.write('\n')