import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv

#here
import re
import requests
from bs4 import BeautifulSoup
page = requests.get("https://weather.com/en-IE/weather/tenday/l/bf217d537cc1c8074ec195ce07fb74de3c1593caa6033b7c3be4645ccc5b01de")
soup = BeautifulSoup(page.content,"lxml")

d = soup.find_all('details', {'id': re.compile(r'^detailIndex')})

for i in d:
    p = i.find('summary')
    print(list(p.stripped_strings))


url = 'https://www.basketball-reference.com/leagues/NBA_2023_standings.html'

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

# Find the table by its class name
#table = soup.find('table', {'class': 'Crom_table__p1iZz'})
table = soup.find('table', {'id': 'confs_standings_E'})
data=[]
# go thru each table and get rows
for row in table.find_all('tr'):
    cols = row.find_all('td')
    x = row.find_all('th')[0].text.strip()
    teamname = x
    if len(cols) > 2:
        win = cols[0].text.strip()

        data.append({'teamname':teamname,'win': win, 'loss': loss,'Points Per Game':pointpergame})

df = pd.DataFrame(data)
df.to_csv('teams.csv', index=False)

##################################imdb

import requests
from bs4 import BeautifulSoup
import re


url = 'https://subslikescript.com/movies'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
ul_scripts_list=soup.find('ul',class_='scripts-list')
for a in ul_scripts_list.find_all('a'):

    print (a.text)


def scrape_movies(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    movie_list = []

    # Each movie is in a 'ul' tag with class 'scripts_list'
    for movie in soup.find_all('ul', class_='scripts-list'):
        # Assuming each movie title is in a 'li' tag
        for title in movie.find_all('li'):
            movie_title = title.get_text()
            movie_list.append(movie_title)

    return movie_list

url = 'https://subslikescript.com/movies'
movies = scrape_movies(url)
for movie in movies:
    print(movie)


#########################################
url = 'https://www.imdb.com/chart/top/'

response = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
soup = BeautifulSoup(response.text, 'html.parser')
movies=soup.find_all('ul',class_='ipc-metadata-list ipc-metadata-list--dividers-between sc-3a353071-0')
for a in ul_scripts_list.find_all('a'):

    print (a.text)

