# NLP Sentiment Analysis on Twitter based on keyword searching

A python project do NLP sentiment analysis on twitter based on keyword searching.

## Introduction

As the social media and internet becoming more and more popular and necessary nowadays, people are expressing their ideas and emotions online more and more frequently. Therefore, it is important to get the sentiment information from people's social media post.  

This project is a combination of web scraping and sentiment analysis using deep learning and neural network to make a sentiment analysis on people's twitter posts. Based on the training outcome of the corpus  `train.csv`, one could get the sentiment analysis on any keyword and on the number of tweets lower than 800 per hour.

## Getting Started

### Prerequisites

First of all sign in your Twitter account and go to [Twitter Apps](https://developer.twitter.com/en/apps). Create a [new app](https://www.toptal.com/python/twitter-data-mining-using-python) and go to Keys and access tokens, copy Consumer Key, Consumer Secret, Access Token and Access Token Secret. Those information will be needed later. 

Using your own Twitter API by calling the classmethod `input_user()` in the python console:

```
> from tweets_download import TweetAnalysis
> sa = TweetAnalysis.input_user()
```
If you are going to use the class more than once, you could change the API by change that in the `Example_Twitter.py`:
```
> sa = TweetAnalysis(consumer_key=your consumer_key,  
consumer_secret=your consumer_secret,  
access_token=your access_token,  
access_token_secret=your access_token_secret)
```
### Installation

Firstly download or clone this repo by the following:
```
$ git clone: https://github.com/mpcs-python/autumn-2019-project-hihowme.git
```
Then navigate to the repo by:
```
$ cd Twitter-Sentimental-Analysis
```
Run the 2 commands below in the terminal to install all the dependencies:
```
$ python3 setup.py install

# download stopwords, punkt and vader_lexicon
$ python3 download.py 

# run this if you have ImportError when running the main document
$ pip install -U scikit-learn
```
## Running the test

To run all the test:
```
$ pytest
```
Run test in a file:
```
python -m pytest -k filenamekeyword (tests matching keyword)
```

To test tweepy cursor [here](http://docs.tweepy.org/en/latest/running_tests.html)
## Built With

-   `Python 3.6`
-   `tweepy`
-   `keras`
-   `gensim`
-   `matplotlib`
-   `tensorflow`
-   `pandas`
-   `nltk`
-   `scikit-learn`

## Code Example

**TwitterAnalysis Class**

To see how this works, navigate to the repo and run the following line in terminal:
```
# Take keyword 'Green day' for instance
$ python3 Example_Twitter.py
Using TensorFlow backend.
Enter Keyword: Green day
Enter Number: 200
The emotion of 200 tweets on Green day. 

General Report: 
negative

Detailed Report: 
23.0 % people thought it was positive
52.5 % people thought it was negative
24.5 % people thought it was neutral
The emotion of 200 tweets on Green day. 

General Report: 
positive

Detailed Report: 
70.5 % people thought it was positive
16.5 % people thought it was negative
13.0 % people thought it was neutral

```
The Figures are showed below:\
![NLTK Example](https://github.com/hihowme/Twitter-Sentimental-Analysis/blob/master/figures/result_1_nltk.png)\
\
![](https://github.com/hihowme/Twitter-Sentimental-Analysis/blob/master/figures/result_2_training.png)

**Training**

The details are shown in the [training_process.ipynb](https://github.com/hihowme/Twitter-Sentimental-Analysis/blob/master/training_process.ipynb), one could change the
data input by rename the csvfile to `train.csv` and copy that into the folder `dataset`.

However, one should notice that the `training.py` is not reusable because the training corpus
is always different. After the training set and testing set is get, one could use the `training.py`
to train a model.

## Training
Though the work of training take a majority part of this project, considering the purpose of this project is the sentiment analysis, I will put the work related to the training in `training_process.ipynb`.

Basically, I used the deep learning method to train a sequential model, using the dataset containing 100,000 original tweets. 
As a result, I got a perfect fitting with the training loss and validation loss converging to each other.
The overall accuracy score is 0.728.

## Limitation & Known Issues

1.  The majority of this work is based on the Twitter API Cursor, which makes it hard to do the test job. However, one could always refers to the test over [here](http://docs.tweepy.org/en/latest/running_tests.html)
2.  Due to the limitation of the Twitter developer API, the maximum number of tweets to analyze is 800. If needed, one could pay a monthly fee to get a premium Twitter API for bigger dataset`.
3.  We include both NLTK analysis and the training method to get a comparison. The NLTK 
sentiment vader should in general be precise, if our model predicts a model that is far
away from the NLTK sentiment result, we should pay attention to what really happened.
4.  This project of the start of the sentiment analysis on the text. In this
training model, due to the corpus bound and algorithm, we only achieved 0.728 accuracy.
In the future, we would try to combining `stochastic search model` in our work
to get some better parameter for the model training.

## Acknowledgments

-   [Stack overflow](https://stackoverflow.com/)
-   Zhi Li  [A Beginner’s Guide to Word Embedding with Gensim Word2Vec Model](https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92)
-   [Keras Tutorial: How to get started with Keras, Deep Learning, and Python](https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/)
-   Javapocalypse  [Twitter Sentiment Analysis in Python Using Tweepy and TextBlob](https://www.youtube.com/watch?v=eFdPGpny_hY)
-   Paolo Ripamonti  [Twitter Sentiment Analysis](https://www.kaggle.com/paoloripamonti/twitter-sentiment-analysis/output)
-   [The Sequential model API](https://keras.io/models/sequential/)
-   [Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory)
-   [Dropout Regularization in Deep Learning Models With Keras](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)
-   Jimit Mistry  [twitter sentiment analysis basic](https://www.kaggle.com/mistryjimit26/twitter-sentiment-analysis-basic)
