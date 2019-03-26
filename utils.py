# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:41:21 2019

@author: Samy Abud Yoshima

"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from tqdm import tqdm, tqdm_notebook
from functools import reduce
import io
from goose3 import Goose

def getSources():
    source_url = 'https://newsapi.org/v1/sources?language=en'
    response = requests.get(source_url).json()
    sources = []
    for source in response['sources']:
        sources.append(source['id'])
    return sources

def mapping():
    d = {}
    response = requests.get('https://newsapi.org/v1/sources?language=en')
    response = response.json()
    for s in response['sources']:
        d[s['id']] = s['category']
    return d

def category(source, m):
    try:
        return m[source]
    except:
        return 'NC'

def getDailyNews():
    sources = getSources()
    key ='7a3123e37aba476bbb4b512f0d9472f7'
    url = 'https://newsapi.org/v1/articles?source={0}&sortBy={1}&apiKey={2}'
    responses = []
    for i, source in tqdm_notebook(enumerate(sources), total=len(sources)):
        
        try:
            u = url.format(source, 'top', key)
        except:
            u = url.format(source, 'latest', key)
        
        response = requests.get(u)
        r = response.json()
        try:
            for article in r['articles']:
                article['source'] = source
            responses.append(r)
        except:
            print('Rate limit exceeded ... please wait and retry in 6 hours')
            return None
                
    articles = list(map(lambda r: r['articles'], responses))
    articles = list(reduce(lambda x,y: x+y, articles))
    
    news = pd.DataFrame(articles)
    news = news.dropna()
    news = news.drop_duplicates()
    news.reset_index(inplace=True, drop=True)
    d = mapping()
    news['category'] = news['source'].map(lambda s: category(s, d))
    news['scraping_date'] = datetime.now()

    try:
        aux = pd.read_csv('./news.csv')
        aux = aux.append(news)
        aux = aux.drop_duplicates('url')
        aux.reset_index(inplace=True, drop=True)
        aux.to_csv('./news.csv', encoding='utf-8', index=False)
    except:
        news.to_csv('./news.csv', index=False, encoding='utf-8')

print('Done getting articles from api.org')
    
if __name__=='__main__':
    getDailyNews()

# Texta database (from running utils.py)
news_list = pd.read_csv("news.csv", encoding='utf-8')
fieldnames = ['author',	'description','publishedAt','source','title','url','urlToImage','category','scraping_date']
urls   = news_list['url'].tolist()      
url_list = pd.DataFrame({'url':urls})
titles = news_list['title'].tolist()      
title_list = pd.DataFrame({'title':titles})
topics = news_list['category'].tolist()
topic_list = pd.DataFrame({'topics':topics})

# 1)Filename: title-<article_numer>.txt, example: title-1.txt
for i in range(0,len(title_list)):
    with io.open("C:\\Users\\Samy Abud Yoshima\\Anaconda3\\Library\\courses\\MIT XPRO\\DataScience+BigData\\Module 1 - Clustering\\CaseSt 1.2.2\\title-" + str(i) + ".txt", 'w', encoding='utf-8') as f:
        title = title_list.iat[int(i),0]
        f.write(title)
        f.write('\n')
# 2) Filename: topic-<article_numer>.txt, example: topic-1.txt
#Contents: The actual “topic” (section or sub-section) under which the news story was classified on the hosting website.
#tag each article with the actual “topic”: assigning the name of the hierarchical identifier for the news story on the website it is hosted on.
for i in range(0,len(topic_list)):
    with io.open("C:\\Users\\Samy Abud Yoshima\\Anaconda3\\Library\\courses\\MIT XPRO\\DataScience+BigData\\Module 1 - Clustering\\CaseSt 1.2.2\\topic-" + str(i) + ".txt", 'w', encoding='utf-8') as f:
        topic = topic_list.iat[int(i),0]
        f.write(topic)
        f.write('\n')
# 2) Filename: article-<article_numer>.txt, example: article-1.txt
#Contents: The contents of the news story.
g = Goose()
for i in range(0,len(url_list)):
    with io.open("C:\\Users\\Samy Abud Yoshima\\Anaconda3\\Library\\courses\\MIT XPRO\\DataScience+BigData\\Module 1 - Clustering\\CaseSt 1.2.2\\article-" + str(i) + ".txt", 'w', encoding='utf-8') as f:
        url= url_list.iat[int(i),0]
        article = g.extract(url=url)
        article = article.cleaned_text
        f.write(article)
        f.write('\n')

print('Done creating files for title, topic and articles')
