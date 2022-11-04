import os
from google.cloud import bigquery

dates = os.listdir('factbook')

client = bigquery.Client()

query = """
SELECT 
date, url, domain, outletName, title
 FROM 
`gdelt-bq.gdeltv2.gal` 
WHERE DATE(date) = "2022-11-04" 
and lang = 'en'
LIMIT 10
"""

query_job = client.query(query)

print("The query data:")
for row in query_job:
    # Row values can be accessed by field name or index.
    print(row)