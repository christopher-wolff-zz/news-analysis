"""Main module."""


__authors__ = 'Christopher Wolff, Kate Chen'
__version__ = '1.0'
__date__ = '9/10/2017'


import json
import urllib

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
# import nltk
# from nltk.tokenize import sent_tokenize
# from pprint import pprint
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
    results = []
    for k in range(num_queries):
        print(f'Running query {k+1}...')
        offset = k * 20
        payload = {'api_key': API_KEY, 'offset': offset}
        response = requests.get(URI, params=payload)
        results += response.json()['results']

    # Save to file
    with open('data/raw.json', 'w') as output_file:
        json.dump(results, output_file)


def analyze():
    """Analyze gathered data."""
    # Calculate sentiment scores
    results_ = json.load(open('data/raw.json'))
    for n, result in enumerate(results_):
        title = result['title']
        abstract = result['abstract']
        # url = result['url']
        '''
        f = urllib.request.urlopen(url)
        soup = BeautifulSoup(f, 'html5lib')
        document = ''
        for par in soup.find_all('p', class_='story-body-text story-content'):
            if par.string:
                document += par.string
        '''
        title_blob = TextBlob(title)
        abstract_blob = TextBlob(abstract)

        title_sent = title_blob.sentiment
        abstract_sent = abstract_blob.sentiment

        result.update({'title_sent': title_sent,
                       'abstract_sent': abstract_sent})
        print(f'{n+1}: {title} -> {title_sent} {abstract_sent}')

    # Save to file
    with open('data/sentiments.json', 'w') as output_file:
        json.dump(results_, output_file)


def visualize():
    """Visualize the data."""
    # Load data
    data = json.load(open('data/sentiments.json'))
    sentiments = [result['title_sent'][0] for result in data]

    # Plot data
    plt.stem(range(1, len(sentiments)+1), sentiments)
    plt.show()


if __name__ == '__main__':
    # query(100)
    # analyze()
    visualize()
