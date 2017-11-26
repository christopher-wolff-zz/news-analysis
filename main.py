"""Main module."""

__authors__ = 'Christopher Wolff, Kate Chen'
__version__ = '1.0'
__date__ = '9/10/2017'

import json
from pprint import pprint
import requests


BASE_URI = 'http://api.nytimes.com/svc/mostpopular/v2'
TYPE = 'mostviewed'
SECTION = 'all-sections'
TIME_PERIOD = '1'
RESPONSE_FORMAT = 'json'


def query(num_queries=1):
    """Request data from NYT and store it as a json file.

    Args:
        num_queries (int): The number of queries

    """
    # Load settings
    data = json.load(open('settings.json'))
    API_KEY = data['API_KEY']

    # Send requests
    URI = f'{BASE_URI}/{TYPE}/{SECTION}/{TIME_PERIOD}.{RESPONSE_FORMAT}'
    results = []
    for k in range(num_queries):
        offset = k * 20
        payload = {'api_key': API_KEY, 'offset': offset}
        response = requests.get(URI, params=payload)
        results += response.json()["results"]

    # Save to file
    with open('data/mostviewed.json', 'w') as output_file:
        json.dump(results, output_file)


class Article():
    """A representation of an article."""

    def __init__(self):
        """Initialize the article."""
        pass


if __name__ == '__main__':
    query(10)
    data = json.load(open('data/mostviewed.json'))
    pprint(len(data))
