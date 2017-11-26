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


def main():
    """Do things."""
    # Load settings
    data = json.load(open('settings.json'))
    API_KEY = data['API_KEY']

    # Send request
    URI = f'{BASE_URI}/{TYPE}/{SECTION}/{TIME_PERIOD}.json?api_key={API_KEY}'
    r = requests.get(URI)
    pprint(r.json())


if __name__ == '__main__':
    main()
