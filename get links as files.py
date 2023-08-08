import requests
from bs4 import BeautifulSoup
import os

# Function to fetch the HTML content of a URL
def get_html_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to fetch URL: {url}")

# Function to extract links from the main page
def extract_links_from_main_page(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []

    for link in soup.find_all('a', href=True):
        links.append(link['href'])

    return links

# Function to download HTML content and save to separate files
def download_html_and_save(links, output_folder):
    count = 1
    for link in links:
        if 'teams' in link:
            if not link.startswith('http'):
                link = 'https://www.basketball-reference.com' + link

            try:
                html_content = get_html_content(link)
                soup = BeautifulSoup(html_content, 'html.parser')

                # Generate a filename based on the link count
                filename = os.path.join("htmlfiles", f"{count}.html")
                count += 1

                # Write the HTML content to a separate file
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(soup.prettify())

                print(f"Downloaded: {link}")
            except Exception as e:
                print(f"Failed to download {link}. Error: {e}")

if __name__ == "__main__":
    main_url = 'https://www.basketball-reference.com/leagues/NBA_2023_standings.html'
    output_folder = 'html_files'  # Folder where HTML files will be saved

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    main_html_content = get_html_content(main_url)
    links = extract_links_from_main_page(main_html_content)
    download_html_and_save(links, output_folder)
