import requests
from bs4 import BeautifulSoup

######################################################## wiki
def fetch_wikipedia_content(url):
    # Get page content
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch page: {response.status_code}")
        return None

    # Parse HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the content section, usually in <div id="bodyContent"> for Wikipedia pages
    content_div = soup.find('div', {'id': 'bodyContent'})
    if not content_div:
        print("Content section not found")
        return None

    # Extract all sections up to "References" and "Notes"
    content_parts = []
    for section in content_div.find_all(['p', 'h2', 'h3']):
        if section.name == 'h2' and section.get_text(strip=True) in ["References", "Notes"]:
            break  # Stop at "References" or "Notes" section
        content_parts.append(section.get_text(strip=True))

    return content_parts


# Example usage
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
content = fetch_wikipedia_content(url)
if content:
    for part in content:
        print(part)

#########################################################get medium article
def fetch_article_until_stop(url):
    # Fetch the page content
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch page: {response.status_code}")
        return None

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Initialize list to hold article content
    article_content = []
    stop_classes = {'pw-multi-vote-count', 'footer'}  # Classes to stop extraction

    # Find the main article section
    main_content = soup.find('article')
    if not main_content:
        print("Article content not found")
        return None

    # Loop through each element in the article until the stop condition is met
    for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'blockquote']):
        # Stop if the element contains any of the stop classes
        if any(cls in stop_classes for cls in element.get('class', [])):
            break
        article_content.append(element.get_text(strip=True))

    return article_content


# Example usage
url = "https://towardsdatascience.com/how-to-evaluate-rag-if-you-dont-have-ground-truth-data-590697061d89"
content = fetch_article_until_stop(url)
if content:
    for part in content:
        print(part)


###################ESPN##################################################
def get_first_espn_article():
    # Step 1: Fetch ESPN's homepage
    homepage_url = "https://www.espn.com/"
    response = requests.get(homepage_url)
    if response.status_code != 200:
        print(f"Failed to fetch homepage: {response.status_code}")
        return None

    # Step 2: Parse the homepage to find the first main article link
    soup = BeautifulSoup(response.text, 'html.parser')
    article_link = None

    # Assuming the first article link is within a 'section' or 'a' tag with a recognizable class
    first_article = soup.find('a', href=True, class_="contentItem__content")  # Class name may vary
    if first_article:
        article_link = "https://www.espn.com" + first_article['href']
    else:
        print("Main article link not found.")
        return None

    # Step 3: Fetch and parse the article page
    article_response = requests.get(article_link)
    if article_response.status_code != 200:
        print(f"Failed to fetch article: {article_response.status_code}")
        return None

    article_soup = BeautifulSoup(article_response.text, 'html.parser')

    # Step 4: Extract main article content, stopping at footer or other unwanted sections
    article_content = []
    main_content = article_soup.find('article')  # Assuming content is within an <article> tag
    if not main_content:
        print("Article content not found.")
        return None

    # Loop through paragraphs and headers to gather the main text
    for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'blockquote']):
        article_content.append(element.get_text(strip=True))

    return article_content


# Example usage
content = get_first_espn_article()
if content:
    for part in content:
        print(part)


###########################CNBC article
def fetch_cnbc_article_content(url):
    # Fetch the article page
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch article: {response.status_code}")
        return None

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Locate the main article content, typically within a specific div or article tag on CNBC
    article_content = []
    main_content = soup.find('div', class_='ArticleBody-articleBody')  # Class may vary based on CNBC's HTML structure
    if not main_content:
        print("Main article content not found")
        return None

    # Extract all paragraph text within the main article content
    for paragraph in main_content.find_all('p'):
        article_content.append(paragraph.get_text(strip=True))

    return article_content


# Example usage
url = "https://www.cnbc.com/2024/10/29/alphabet-to-report-q3-earnings-after-the-bell.html"
content = fetch_cnbc_article_content(url)
if content:
    for part in content:
        print(part)



#########################################SEEKING ALPHA article

def fetch_seeking_alpha_article_content(url):
    # Fetch the article page
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch article: {response.status_code}")
        return None

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Locate the main article content
    article_content = []
    main_content = soup.find('div',
                             class_='content-article')  # Class may vary depending on Seeking Alpha's HTML structure
    if not main_content:
        print("Main article content not found")
        return None

    # Extract all paragraph text within the main article content
    for paragraph in main_content.find_all('p'):
        article_content.append(paragraph.get_text(strip=True))

    return article_content


# Example usage
url = "https://seekingalpha.com/news/4220953-key-takeaways-from-alphabets-q3-earnings-as-stock-rises-4"
content = fetch_seeking_alpha_article_content(url)
if content:
    for part in content:
        print(part)