import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
#https://www.google.com/search?q=nba+standings&rlz=1C1RXQR_enCA932CA932&oq=nba+standings&gs_lcrp=EgZjaHJvbWUqDQgAEAAYgwEYsQMYgAQyDQgAEAAYgwEYsQMYgAQyBggBEAAYAzINCAIQABiDARixAxiABDINCAMQABiDARixAxiABDINCAQQABiDARixAxiABDINCAUQABiDARixAxiABDIGCAYQRRg8MgYIBxBFGDzSAQgxNTk5ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8#sie=lg;/g/11snv1vp6v;3;/m/05jvx;st;fp;1;;;
### this extracts the table from the page https://www.basketball-reference.com/leagues/NBA_2023_standings.html
## put each table on the page in a csv file

# Function to fetch the HTML content of a URL 33 1 werwer
def get_html_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to fetch URL: {url}")



# extract tables from the main page
def extract_tables_from_main_page(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')
    return tables

# save the table as a CSV file
def save_table_as_csv(table, filename):
    df = pd.read_html(str(table))[0]
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    main_url = 'https://www.basketball-reference.com/leagues/NBA_2023_standings.html'
    #main_url = 'https://www.basketball-reference.com/leagues/NBA_2023_per_game.html'

    output_folder = 'csv_files'  # Folder where CSV files will be saved

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    main_html_content = get_html_content(main_url)
    tables = extract_tables_from_main_page(main_html_content)

    count = 1
    for table in tables:
        filename = os.path.join("csv files", f"{count}.csv")
        save_table_as_csv(table, filename)
        print(f"Saved table {count} as CSV: {filename}")
        count += 1
