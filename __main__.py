"""Sample use of analyzer module."""

from analyzer import query, scrape_stories


if __name__ == '__main__':
    print('Requesting articles from NYT API')
    print('================================')
    query(num_queries=10)
    print()

    print('Scraping article texts from NYT website')
    print('=======================================')
    scrape_stories()
    print()

    print('Training classifiers')
    print('====================')
    train_model()
