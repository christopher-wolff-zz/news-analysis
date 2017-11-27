"""Main module."""


__authors__ = 'Christopher Wolff, Kate Chen'
__version__ = '1.0'
__date__ = '9/10/2017'


import json
import urllib

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
# import nltk
# from nltk.tokenize import sent_tokenize
import requests
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


BASE_URI = 'http://api.nytimes.com/svc/mostpopular/v2'
TYPE = 'mostviewed'
SECTION = 'all-sections'
TIME_PERIOD = '30'
RESPONSE_FORMAT = 'json'


def query(num_queries=1):
    """Request data from NYT and store it as a json file.

    Args:
        num_queries (int): The number of queries

    """
    # Load settings
    settings = json.load(open('settings.json'))
    API_KEY = settings['API_KEY']
    # Send requests
    URI = f'{BASE_URI}/{TYPE}/{SECTION}/{TIME_PERIOD}.{RESPONSE_FORMAT}'
    articles = []
    for k in range(num_queries):
        print(f'Running query {k+1}...')
        offset = k * 20
        payload = {'api_key': API_KEY, 'offset': offset}
        response = requests.get(URI, params=payload)
        articles += response.json()['results']
    # Save to file
    with open('data/raw.json', 'w') as output_file:
        json.dump(articles, output_file)


def get_stories():
    """Get full document texts from urls."""
    articles = json.load(open('data/raw.json'))
    for k, article in enumerate(articles):
        print(f'Analyzing article {k+1}...')
        url = article['url']
        f = urllib.request.urlopen(url)
        soup = BeautifulSoup(f, 'html5lib')
        story = ''
        for par in soup.find_all('p', class_='story-body-text story-content'):
            if par.string:
                story += ' ' + par.string
        article.update({'story': story})
    # Save to file
    with open('data/sentiments.json', 'w') as output_file:
        json.dump(articles, output_file)


def analyze():
    """Analyze gathered data."""
    # Calculate sentiment scores
    articles = json.load(open('data/stories.json'))
    for k, article in enumerate(articles):
        print(f'Analyzing article {k+1}...')
        title = article['title']
        abstract = article['abstract']
        story = article['story']

        title_blob = TextBlob(title, analyzer=NaiveBayesAnalyzer())
        abstract_blob = TextBlob(abstract, analyzer=NaiveBayesAnalyzer())
        story_blob = TextBlob(story, analyzer=NaiveBayesAnalyzer())

        title_sent = title_blob.sentiment
        abstract_sent = abstract_blob.sentiment
        story_sent = story_blob.sentiment

        article.update({'title_sent': title_sent,
                        'abstract_sent': abstract_sent,
                        'story_sent': story_sent})

        print(f'{k+1}: {title} {title_sent.p_pos} {abstract_sent.p_pos} \
              {story_sent.p_pos}')

    # Save to file
    with open('data/sentiments.json', 'w') as output_file:
        json.dump(articles, output_file)


def visualize():
    """Visualize the data."""
    # Load data
    articles = json.load(open('data/sentiments.json'))
    title_sents = [article['title_sent'][0] for article in articles]
    abstract_sents = [article['abstract_sent'][0] for article in articles]
    story_sents = [article['story_sent'][0] for article in articles]

    # Plot data
    plt.figure(1)
    plt.stem(range(1, len(title_sents)+1), title_sents)
    plt.title('Title Sentiment')
    plt.xlabel('Rank')
    plt.ylabel('Sentiment')
    plt.figure(2)
    plt.stem(range(1, len(abstract_sents)+1), abstract_sents)
    plt.title('Abstract Sentiment')
    plt.xlabel('Rank')
    plt.ylabel('Sentiment')
    plt.figure(3)
    plt.stem(range(1, len(story_sents)+1), story_sents)
    plt.title('Story Sentiment')
    plt.xlabel('Rank')
    plt.ylabel('Sentiment')
    plt.show()


if __name__ == '__main__':
    # query(100)
    # get_stories()
    analyze()
    # visualize()
