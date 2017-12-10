# News Analysis

A framework for automated web scraping and sentiment analysis for news articles. Provides trained SVM and Naive Bayes classifiers.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This project requires python version >= 3.5, which can be installed from [here](https://www.python.org/downloads/).

### Installing

To download the project source directory, simply open terminal and run
```
git clone git@github.com:christopher-wolff/news-analysis.git
```
or download it directly from GitHub.

(optional) We highly recommend using a python virtual environment to install dependencies. Creating the environment is as simple as
```
cd /path/to/project/directory/
python3 -m venv virtualenv
source virtualenv/bin/activate
```
More information about python virtual environments can be found [here](https://virtualenv.pypa.io/en/stable/userguide/).

This project has the following dependencies:
* [bs4](https://www.crummy.com/software/BeautifulSoup/)
* [matplotlib](https://matplotlib.org/)
* [nltk](http://www.nltk.org/)
* [requests](http://docs.python-requests.org/en/master/)
* [scipy](https://www.scipy.org/)
* [sklearn](http://scikit-learn.org/)
* [textblob](https://pypi.python.org/pypi/textblob)

These can be installed using the provided requirements.txt with the command
```
pip3 install -r requirements.txt
```
inside the project root directory.

Lastly, the nltk library requires downloading all needed files. This can be done by launching an interactive python shell with
```
python3
```
and then running
```
import nltk
nltk.download('stopwords')
```
You may need to run
```
/Applications/Python\ 3.6/Install\ Certificates.command
```
prior to that to resolve a certificate conflict that appears between nltk and python 3.

In order to use the *query* feature that enables you to send requests to the New York Times Most Popular API and store the results automatically, you will need to obtain an API key [here](https://developer.nytimes.com/signup). Add it to file *settings_example.json* under the "API" field, and rename the file to *settings.json*.

## Running the project

The project can be executed from a python shell with
```
python3
from analyzer import \*
```

All methods in this module operate independent of each other, which means that the shell can safely be closed after executing any one of them. All intermediate results are stored in separate files within the *data* folder.

### Break down of available methods
#### query
The **query** method allows you to automatically send API calls to the New York Times Most Popular API. These results will be stored in a file called *raw.json*.

#### scrape_stories
The returned API calls only contain the article title and abstract, but not the full document story. The **scrape_stories** method uses the *beautifulsoup* module to scrape the story content for all articles from the official New York Times website and stores it in a file called *with_stories.json*.

#### label_articles
**label_articles** is a UI that can be used to manually label the returned articles and was originally used to obtain training data for the sentiment classifiers. More information can be found in the method docstring.

#### train_model
This method trains two different sentiment classifiers: a support vector machine and a multinomial naive bayes classifier, using the *sklearn* library. The documents can be modeled using either a *term frequency - inverse document frequency* vectorizer or a *bag of words* model. The resulting models are dumped into the /models/ folder using the python object serialization tool *pickle* and can easily be extracted.

#### analyze
**analyze** was originally used to analyze sentiment using the *TextBlob* library. However, we eventually moved on towards training our own classifier. This method is included anyway in case it might be useful to someone.

#### visualize
The **visualize** method plots the results using *matplotlib*.

## Deployment

The following snippet shows how you might use the trained classifiers on your own documents:
```
import pickle
import sklearn
clf = pickle.load(open('models/mnb', 'rb'))
```
The resulting *clf* object will be a trained sklearn classifier, whose usage instructions can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [here](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
