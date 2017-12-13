"""Sample use of analyzer module."""

from analyzer import query
from analyzer import scrape_stories
from analyzer import label_articles
from analyzer import train_model
from analyzer import analyze
from analyzer import visualize

if __name__ == '__main__':
    print('Requesting articles from NYT API')
    print('================================')
    query(num_queries=1)
    print()

    print('Scraping full article texts from NYT website')
    print('============================================')
    scrape_stories()
    print()

    print('Labeling articles')
    print('=================')
    label_articles(reset=True, rand_labels=True)
    print()

    print('Training classifiers')
    print('====================')
    train_model()
    print()

    print('Analyzing data')
    print('====================')
    analyze()
    print()

    print('Visualize results')
    print('=================')
    visualize()
