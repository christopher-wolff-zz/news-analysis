"""Main module."""


__authors__ = 'Christopher Wolff, Kate Chen'
__version__ = '1.0'
__date__ = '9/10/2017'


import json
import os.path
import pickle
import random
import urllib

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn import svm
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import matplotlib.pyplot as plt
import requests


SETTINGS_PATH = 'settings.json'
RAW_PATH = 'data/raw.json'
STORIES_PATH = 'data/with_stories.json'
LABELS_PATH = 'data/with_labels.json'
SENTIMENTS_PATH = 'data/with_sentiments.json'
MNB_PATH = 'models/mnb'
SVM_PATH = 'models/svm'

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


def label_articles(reset=False, relabel=False, start=0, rand_labels=False):
    """Run UI for sentiment labeling.

    Loads all articles and presents those without a label.

    Args:
        reset (boolean): Delete all labels
        relabel (boolean): Allow option to override existing labels
        start (int): Article number to start from
        rand_labels (boolean): Assign all random labels
    """
    # Load articles
    if reset or not os.path.isfile(LABELS_PATH):
        articles = json.load(open(STORIES_PATH))
    else:
        articles = json.load(open(LABELS_PATH))
    if start >= len(articles):
        raise ValueError(f'Invalid starting point: {start}')
    # Label articles
    for k, article in enumerate(articles[start:]):
        if not relabel and 'sentiment' in article:
            continue
        print(f'Article: {k+start+1}')
        print(f"Title: {article['title']}")
        print(f"Abstract: {article['abstract']}")
        sentiments = [-1, 1]
        if rand_labels:
            sent = random.choice(sentiments)
        else:
            try:
                sent = int(input('Label: '))
            except ValueError:
                break
            if sent not in sentiments:
                break
        article.update({'sentiment': sent})
        print('----------------------------')
    # Save articles
    with open(LABELS_PATH, 'w') as output_file:
        json.dump(articles, output_file)


def train_model(vect='tfidf', random_state=None):
    """Train a sentiment analyzer model.

    Args:
        vect (str): The method used to convert input text to a vector
                    'count' -> Bag of words
                    'tfidf' -> Term frequency - Inverse document frequency
        random_state (int): Random seed for train_test_split used by numpy
    """
    # Load articles
    articles = json.load(open(LABELS_PATH))
    # Extract data
    articles = [article for article in articles if 'sentiment' in article]
    stopset = set(stopwords.words('english'))
    stories = [article['story'] for article in articles]
    labels = [article['sentiment'] for article in articles]
    # Vectorize data
    if vect == 'tfidf':
        vectorizer = TfidfVectorizer(use_idf=True,
                                     lowercase=True,
                                     strip_accents='ascii',
                                     stop_words=stopset)
    else:
        vectorizer = CountVectorizer(lowercase=True,
                                     strip_accents='ascii',
                                     stop_words=stopset)
    x = vectorizer.fit_transform(stories)
    y = labels
    if random_state is not None:
        x_train, x_test, y_train, y_test = train_test_split(
                x, y, random_state=random_state)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y)
    # Analyze and display relevant information
    num_total = len(articles)
    num_pos = sum(article['sentiment'] == 1 for article in articles)
    num_neg = sum(article['sentiment'] == -1 for article in articles)
    percent_train = round(len(y_train) / num_total * 100)
    percent_test = round(len(y_test) / num_total * 100)
    print(f'Found {num_total} labeled articles')
    print(f'{num_pos} positive, {num_neg} negative')
    print(f'{percent_train}% train, {percent_test}% test')
    # Train multinomial naive bayes classifier and evaluate its accuracy
    mnb_clf = naive_bayes.MultinomialNB()
    mnb_clf.fit(x_train, y_train)
    y_pred = mnb_clf.predict(x_test)
    mnb_acc = accuracy_score(y_test, y_pred)
    print(f'MNB: {mnb_acc}')
    # Train support vector machine and evaluate its accuracy
    svm_clf = svm.SVC(probability=True)
    svm_clf.fit(x_train, y_train)
    y_pred = svm_clf.predict(x_test)
    svm_acc = accuracy_score(y_test, y_pred)
    print(f'SVM: {svm_acc}')
    # Store trained classifiers
    with open(SVM_PATH, 'wb') as output_file:
        pickle.dump(mnb_clf, output_file)
    with open(MNB_PATH, 'wb') as output_file:
        pickle.dump(svm_clf, output_file)


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


def visualize():
    """Visualize the data."""
    # Load data
    articles = json.load(open(LABELS_PATH))
    # title_sents = [article['title_sent'][0] for article in articles]
    # abstract_sents = [article['abstract_sent'][0] for article in articles]
    # story_sents = [article['story_sent'][0] for article in articles]

    neg_avg = []
    total_neg = 0
    total = 0
    for k, a in enumerate(articles):
        if 'sentiment' not in a:
            continue
        if a['sentiment'] == -1:
                total_neg += 1
        neg_avg.append(total_neg / (k+1))
        total += 1

    # avg = np.array([total_neg / total for i in range(total)])
    plt.figure(1)
    plt.plot(range(1, total + 1), neg_avg)
    # plt.plot(range(1, total + 1), avg)
    plt.title('Cumulative Average of Negative Articles')
    plt.xlabel('Popularity')
    plt.ylabel('% Negative')
    """
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
    """

    plt.show()
