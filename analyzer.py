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
import numpy as np


SETTINGS_PATH = 'settings.json'
RAW_PATH = 'data/raw.json'
STORIES_PATH = 'data/with_stories.json'
LABELS_PATH = 'data/with_labels.json'
SENTIMENTS_PATH = 'data/with_sentiments.json'
MNB_PATH = 'models/mnb.pkl'
SVM_PATH = 'models/svm.pkl'
COUNT_VECT_PATH = 'models/count_vect.pkl'
TFIDF_VECT_PATH = 'models/tfidf_vect.pkl'

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
        print(f'Scraping article {k+1}...')
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
    sentiments = [-1, 1]
    print(f'Available sentiments: {sentiments}')
    for k, article in enumerate(articles[start:]):
        if not relabel and 'sentiment' in article:
            continue
        print(f'Article: {k+start+1}')
        print(f"Title: {article['title']}")
        print(f"Abstract: {article['abstract']}")
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


def train_model(random_state=None):
    """Train a sentiment analyzer model.

    Args:
        random_state (int): Random seed for train_test_split used by numpy
    """
    # Load articles
    articles = json.load(open(LABELS_PATH))
    # Extract data
    articles = [article for article in articles if 'sentiment' in article]
    stopset = set(stopwords.words('english'))
    titles = [article['title'] for article in articles]
    labels = [article['sentiment'] for article in articles]

    # Vectorize data
    count_vect = CountVectorizer(lowercase=True,
                                 strip_accents='ascii',
                                 stop_words=stopset,
                                 decode_error='replace')
    tfidf_vect = TfidfVectorizer(use_idf=True,
                                 lowercase=True,
                                 strip_accents='ascii',
                                 stop_words=stopset,
                                 decode_error='replace')

    # Analyze and display relevant information
    num_total = len(articles)
    num_pos = sum(article['sentiment'] == 1 for article in articles)
    num_neg = sum(article['sentiment'] == -1 for article in articles)
    print(f'Found {num_total} labeled articles')
    print(f'{num_pos} +, {num_neg} -')

    # Train using count vectorizer
    print('Vectorizing using bag of words...')
    x = count_vect.fit_transform(titles)
    y = labels
    if random_state is not None:
        x_train, x_test, y_train, y_test = train_test_split(
                x, y, random_state=random_state)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y)

    mnb_clf = naive_bayes.MultinomialNB()
    mnb_clf.fit(x_train, y_train)
    y_pred = mnb_clf.predict(x_test)
    mnb_acc = accuracy_score(y_test, y_pred) * 100
    print('Naive Bayes: %.2f%% accuracy' % mnb_acc)

    svm_clf = svm.SVC(probability=True)
    svm_clf.fit(x_train, y_train)
    y_pred = svm_clf.predict(x_test)
    svm_acc = accuracy_score(y_test, y_pred) * 100
    print('SVM: %.2f%% accuracy' % svm_acc)

    # Train using tfidf vectorizer
    print('Vectorizing using tfidf...')
    x = tfidf_vect.fit_transform(titles)
    y = labels
    if random_state is not None:
        x_train, x_test, y_train, y_test = train_test_split(
                x, y, random_state=random_state)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y)

    mnb_clf = naive_bayes.MultinomialNB()
    mnb_clf.fit(x_train, y_train)
    y_pred = mnb_clf.predict(x_test)
    mnb_acc = accuracy_score(y_test, y_pred) * 100
    print('Naive Bayes: %.2f%% accuracy' % mnb_acc)

    svm_clf = svm.SVC(probability=True)
    svm_clf.fit(x_train, y_train)
    y_pred = svm_clf.predict(x_test)
    svm_acc = accuracy_score(y_test, y_pred) * 100
    print('SVM: %.2f%% accuracy' % svm_acc)

    # Store vectorizers and trained classifiers
    with open(SVM_PATH, 'wb') as output_file:
        pickle.dump(mnb_clf, output_file)
    with open(MNB_PATH, 'wb') as output_file:
        pickle.dump(svm_clf, output_file)
    with open(COUNT_VECT_PATH, 'wb') as output_file:
        pickle.dump(count_vect.vocabulary_, output_file)
    with open(TFIDF_VECT_PATH, 'wb') as output_file:
        pickle.dump(tfidf_vect.vocabulary_, output_file)


def analyze():
    """Analyze article data."""
    # Calculate sentiment scores
    articles = json.load(open(LABELS_PATH))
    mnb_clf = pickle.load(open(MNB_PATH, 'rb'))
    svm_clf = pickle.load(open(SVM_PATH, 'rb'))
    count_vocabulary = pickle.load(open(COUNT_VECT_PATH, 'rb'))
    tfidf_vocabulary = pickle.load(open(TFIDF_VECT_PATH, 'rb'))
    stopset = set(stopwords.words('english'))
    count_vect = CountVectorizer(lowercase=True,
                                 strip_accents='ascii',
                                 stop_words=stopset,
                                 decode_error='replace',
                                 vocabulary=count_vocabulary)
    tfidf_vect = TfidfVectorizer(use_idf=True,
                                 lowercase=True,
                                 strip_accents='ascii',
                                 stop_words=stopset,
                                 decode_error='replace',
                                 vocabulary=tfidf_vocabulary)
    for k, article in enumerate(articles):
        title = article['title']
        abstract = article['abstract']
        story = article['story']
        print(f'{k+1}: {title}')
        title_sent = TextBlob(title).sentiment
        abstract_sent = TextBlob(abstract).sentiment
        story_sent = TextBlob(story).sentiment
        article.update({'title_sent': title_sent,
                        'abstract_sent': abstract_sent,
                        'story_sent': story_sent})
        print(f'{title_sent} {abstract_sent} {story_sent}')

        count = count_vect.fit_transform([title])
        tfidf = tfidf_vect.fit_transform([title])
        article.update({'count_mnb_sent': mnb_clf.predict(count).item(0),
                        'count_svm_sent': svm_clf.predict(count).item(0),
                        'tfidf_mnb_sent': mnb_clf.predict(tfidf).item(0),
                        'tfidf_svm_sent': svm_clf.predict(tfidf).item(0)})

    # Test TextBlob performance
    num_total = 0
    num_correct = 0
    for article in articles:
        if 'sentiment' not in article:
            continue
        title_sent = article['title_sent'].polarity
        true_sent = article['sentiment']
        if title_sent == 0:
            continue
        if _sign(title_sent) == true_sent:
            num_correct += 1
        num_total += 1
    acc = num_correct / num_total * 100
    print('=========================')
    print('TextBlob accuracy: %.2f' % acc)
    print('=========================')

    # Determine min, max, mean, and std
    title_sents = np.array([a['title_sent'] for a in articles])
    abstract_sents = np.array([a['abstract_sent'] for a in articles])
    story_sents = np.array([a['story_sent'] for a in articles])

    print('Title Sentiments')
    print('----------------')
    print(f'min: {np.min(title_sents)}')
    print(f'max: {np.max(title_sents)}')
    print(f'mean: {np.mean(title_sents)}')
    print(f'std: {np.std(title_sents)}')
    print()

    print('Abstract Sentiments')
    print('-------------------')
    print(f'min: {np.min(abstract_sents)}')
    print(f'max: {np.max(abstract_sents)}')
    print(f'mean: {np.mean(abstract_sents)}')
    print(f'std: {np.std(abstract_sents)}')
    print()

    print('Story Sentiments')
    print('----------------')
    print(f'min: {np.min(story_sents)}')
    print(f'max: {np.max(story_sents)}')
    print(f'mean: {np.mean(story_sents)}')
    print(f'std: {np.std(story_sents)}')
    print()

    # Save to file
    with open(SENTIMENTS_PATH, 'w') as output_file:
        json.dump(articles, output_file)


def visualize():
    """Visualize the data."""
    # Load data
    articles = json.load(open(SENTIMENTS_PATH))
    title_sents = [article['title_sent'][0] for article in articles]
    abstract_sents = [article['abstract_sent'][0] for article in articles]
    story_sents = [article['story_sent'][0] for article in articles]
    count_mnb_sents = [article['count_mnb_sent'] for article in articles]
    count_svm_sents = [article['count_svm_sent'] for article in articles]
    tfidf_mnb_sents = [article['tfidf_mnb_sent'] for article in articles]
    tfidf_svm_sents = [article['tfidf_svm_sent'] for article in articles]

    view_rank = range(1, len(articles) + 1)

    # Calculate trendlines
    z1 = np.polyfit(view_rank, title_sents, 1)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(view_rank, abstract_sents, 1)
    p2 = np.poly1d(z2)
    z3 = np.polyfit(view_rank, story_sents, 1)
    p3 = np.poly1d(z3)

    z4 = np.polyfit(view_rank, count_mnb_sents, 1)
    p4 = np.poly1d(z4)
    z5 = np.polyfit(view_rank, count_svm_sents, 1)
    p5 = np.poly1d(z5)
    z6 = np.polyfit(view_rank, tfidf_mnb_sents, 1)
    p6 = np.poly1d(z6)
    z7 = np.polyfit(view_rank, tfidf_svm_sents, 1)
    p7 = np.poly1d(z7)

    # Compute moving average
    window_size = 10
    window = np.ones(int(window_size))/float(window_size)
    count_svm_sents_ma = np.convolve(count_svm_sents, window, 'same')
    tfidf_svm_sents_ma = np.convolve(tfidf_svm_sents, window, 'same')

    # Plot sentiment versus view rank
    # TextBlob
    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.scatter(view_rank, title_sents, s=5)
    plt.plot(view_rank, p1(view_rank), 'r--')
    plt.title('Title Sentiment')
    plt.xlabel('View Rank')
    plt.ylabel('Sentiment Score')
    plt.ylim(-1.1, 1.1)

    plt.subplot(1, 3, 2)
    plt.scatter(view_rank, abstract_sents, s=5)
    plt.plot(view_rank, p2(view_rank), 'r--')
    plt.title('Abstract Sentiment')
    plt.xlabel('View Rank')
    plt.ylim(-1.1, 1.1)

    plt.subplot(1, 3, 3)
    plt.scatter(view_rank, story_sents, s=5)
    plt.plot(view_rank, p3(view_rank), 'r--')
    plt.title('Story Sentiment')
    plt.xlabel('View Rank')
    plt.ylim(-1.1, 1.1)

    # sklearn classifiers
    plt.figure(2)
    plt.subplot(2, 2, 1)
    plt.scatter(view_rank, count_mnb_sents, s=5)
    plt.plot(view_rank, p4(view_rank), 'r--')
    plt.title('Bag of Words + Naive Bayes')
    plt.ylabel('Sentiment Score')
    plt.ylim(-1.1, 1.1)

    plt.subplot(2, 2, 2)
    plt.scatter(view_rank, count_svm_sents, s=5)
    plt.scatter(view_rank, count_svm_sents_ma, s=5, facecolor='0.5')
    plt.plot(view_rank, p5(view_rank), 'r--')
    plt.title('Bag of Words + SVM')
    plt.ylim(-1.1, 1.1)

    plt.subplot(2, 2, 3)
    plt.scatter(view_rank, tfidf_mnb_sents, s=5)
    plt.plot(view_rank, p6(view_rank), 'r--')
    plt.title('Tfidf + Naive Bayes')
    plt.xlabel('View Rank')
    plt.ylabel('Sentiment Score')
    plt.ylim(-1.1, 1.1)

    plt.subplot(2, 2, 4)
    plt.scatter(view_rank, tfidf_svm_sents, s=5)
    plt.scatter(view_rank, tfidf_svm_sents_ma, s=5, facecolor='0.5')
    plt.plot(view_rank, p7(view_rank), 'r--')
    plt.title('Tfidf + SVM')
    plt.xlabel('View Rank')
    plt.ylim(-1.1, 1.1)

    plt.show()


def _sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0
