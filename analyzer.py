"""Main module."""


__authors__ = 'Christopher Wolff, Kate Chen'
__version__ = '1.0'
__date__ = '9/10/2017'


import json
import os.path
import urllib

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
# import nltk
# from nltk.tokenize import sent_tokenize
import requests
from textblob import TextBlob


SETTINGS_PATH = 'settings.json'
RAW_PATH = 'data/raw.json'
STORIES_PATH = 'data/with_stories.json'
LABELS_PATH = 'data/with_labels.json'
SENTIMENTS_PATH = 'data/with_sentiments.json'

BASE_URI = 'http://api.nytimes.com/svc/mostpopular/v2'
TYPE = 'mostviewed'
SECTION = 'all-sections'
TIME_PERIOD = '1'
RESPONSE_FORMAT = 'json'


class Analyzer():
    """A news analyzer.

    Provides tools for running API queries, web scraping, sentiment analysis,
    and data visualization.

    """

    @staticmethod
    def query(num_queries=1):
        """Request data from NYT and store it as a json file.

        Args:
            num_queries (int): The number of queries

        """
        # Load API key
        settings = json.load(open(SETTINGS_PATH))
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
        with open(RAW_PATH, 'w') as output_file:
            json.dump(articles, output_file)

    @staticmethod
    def scrape_stories():
        """Get full document texts from urls."""
        # Load articles
        articles = json.load(open(RAW_PATH))
        # Submit GET request and parse response content
        for k, article in enumerate(articles):
            print(f'Analyzing article {k+1}...')
            url = article['url']
            f = urllib.request.urlopen(url)
            soup = BeautifulSoup(f, 'html5lib')
            story = ''
            for par in soup.find_all('p', class_='story-body-text \
                                                  story-content'):
                if par.string:
                    story += ' ' + par.string
            article.update({'story': story})
        # Save articles
        with open(STORIES_PATH, 'w') as output_file:
            json.dump(articles, output_file)

    @staticmethod
    def label(reset=False):
        """Run UI for sentiment labeling.

        Args:
            reset (boolean): Reset all labels if true

        """
        # Load articles
        if reset or not os.path.isfile(LABELS_PATH):
            articles = json.load(open(STORIES_PATH))
        else:
            articles = json.load(open(LABELS_PATH))
        # Label articles
        for k, article in enumerate(articles):
            if 'sentiment' in article:
                continue
            print(f"Article: {k+1}")
            print(f"Title: {article['title']}")
            print(f"Abstract: {article['abstract']}")
            sentiment_labels = [-1, 0, 1]
            try:
                label = int(input('Label: '))
            except ValueError:
                break
            if label not in sentiment_labels:
                break
            article.update({'sentiment': label})
            print('----------------------------')
        # Save articles
        with open(LABELS_PATH, 'w') as output_file:
            json.dump(articles, output_file)

    @staticmethod
    def analyze():
        """Analyze gathered data."""
        # Calculate sentiment scores
        articles = json.load(open(STORIES_PATH))
        for k, article in enumerate(articles):
            title = article['title']
            abstract = article['abstract']
            story = article['story']

            print(f'{k+1}: {title}')
            title_blob = TextBlob(title)
            abstract_blob = TextBlob(abstract)
            story_blob = TextBlob(story)

            title_sent = title_blob.sentiment
            abstract_sent = abstract_blob.sentiment
            story_sent = story_blob.sentiment

            article.update({'title_sent': title_sent,
                            'abstract_sent': abstract_sent,
                            'story_sent': story_sent})

            print(f'{title_sent} {abstract_sent} {story_sent}')

        # Save to file
        with open(SENTIMENTS_PATH, 'w') as output_file:
            json.dump(articles, output_file)

    @staticmethod
    def visualize():
        """Visualize the data."""
        # Load data
        articles = json.load(open(SENTIMENTS_PATH))
        title_sents = [article['title_sent'][0] for article in articles]
        abstract_sents = [article['abstract_sent'][0] for article in articles]
        # story_sents = [article['story_sent'][0] for article in articles]

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

        '''
        plt.figure(3)
        plt.stem(range(1, len(story_sents)+1), story_sents)
        plt.title('Story Sentiment')
        plt.xlabel('Rank')
        plt.ylabel('Sentiment')
        '''

        plt.show()
