# Tweet analysis
import tweepy
import os
import pandas as pd
from preprocessing import clean_text, share
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentiment_analysis import sentiment_analysis
import matplotlib.pyplot as plt


class TweetAnalysis:
    """
        A class used to scrape and do sentiment analysis on twitter based on keyword searching

        ...

        Attributes
        ----------
        keyword : str
            The keyword you want to search
        num : int
            The number of twitter post you want to analyze
        path : string
            A path to the csv file where the data would be downloaded
        text : list
            A empty list used to save the text of tweets for download
        sentiment_vader : list
            A empty list used to save the sentiment analyzed by nltk vader
        sentiment_ : list
            A empty list used to save the sentiment analyzed by training model


        Methods
        -------
        bar_plot(*args, training)
            Return the bar plot of the result
        download_data():
            Download all the tweets text in a csv file
        get_sentiment_nltk()
            Use nltk vader to analyze sentiment on tweet post and return different groups of sentiment
        sentiment_training()
            Use training model to analyze sentiment on tweet post and return different groups of sentiment
        get_text()
            get the cleaned text from api and store that in text attribute
        show_report(*args， training)
            print the result of sentiment analysis
        input_user(cls):
            setting your own Twitter API
    """

    def __init__(
        self,
        consumer_key="yDedg5opOjR66byf1Kl4DAkHP",
        consumer_secret="SThpIlJDJPisRZ5q8W1XkDXXoTQUN3Uz8HK83a8EFPpEFLQBvN",
        access_token="979582225531351041-6PONTVxrM1MdmQ5l9B12BPV6mkj9NaP",
        access_token_secret="2VbZLvhAjz9ZVEjpx9W9EeAT7wDvg2GRC2HG3xVjx2Npq",
        keyword="Beatles",
        num=200,
        path="sentiment_data.csv",
    ):
        self.path = path
        self.keyword = keyword
        self.num = num

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)

        self.text = []
        self.sentiment_vader = []
        self.sentiment_ = []

    def get_text(self):
        """

        get the cleaned text from api and store that in text attribute

        """
        temp_tweets = tweepy.Cursor(self.api.search, q=self.keyword, lang="en").items(
            self.num
        )
        for tweets in temp_tweets:
            self.text.append(clean_text(tweets.text))

    def get_sentiment_nltk(self):
        """Use nltk vader to analyze sentiment on tweet post and return different groups of sentiment

        Returns
        -------
        tuple
            a tuple of strings representing the percentage of different groups in result using TextBlob
        """
        s_nltk = SentimentIntensityAnalyzer()

        polarity = 0
        positive = 0
        negative = 0
        neutral = 0
        n = self.num
        key = self.keyword

        for text in self.text:
            p = s_nltk.polarity_scores(text)["compound"]
            polarity += p
            self.sentiment_vader.append(p)
            if p < 0:
                negative += 1
            elif p == 0:
                neutral += 1
            elif p > 0:
                positive += 1

        positive = share(positive, n)
        negative = share(negative, n)
        neutral = share(neutral, n)

        polarity = share(polarity, n) / 100

        return (
            polarity,
            positive,
            negative,
            neutral,
            key,
            n,
        )

    def sentiment_training(self):
        """Use training model to analyze sentiment on tweet post and return different groups of sentiment

        Returns
        -------
        tuple
            a tuple of strings representing the percentage of different groups in result using Training
        """

        polarity = 0
        positive = 0
        negative = 0
        neutral = 0
        n = self.num
        key = self.keyword

        for text in self.text:
            p = sentiment_analysis(text)
            polarity += p
            self.sentiment_.append(p)
            if p <= 0.56:
                negative += 1
            elif 0.56 < p <= 0.67:
                neutral += 1
            elif p > 0.67:
                positive += 1

        positive = share(positive, n)
        negative = share(negative, n)
        neutral = share(neutral, n)

        polarity = share(polarity, n) / 100

        return (
            polarity,
            positive,
            negative,
            neutral,
            key,
            n,
        )

    # def sentiment_analysis(self):

    def bar_plot(
        self, polarity, positive, negative, neutral, keyword, num, training=True
    ):
        """Return the bar plot of the result

        Parameters
        ----------
        positive, negative, neutral: str
            The percentage of different groups in final result
        keyword : str
            The keyword you want to search
        num : int
            The number of twitter post you want to analyze
        training: Boolean
            True would return the graph of the training model analysis,
            False would return the graph of the vader method

        Returns
        -------
        plt bar plot
            a bar plot of the result
        """

        s = pd.Series([negative, neutral, positive], index=["NEG", "NEU", "POS"],)

        s = s.astype(float)

        # Set descriptions:
        if training:
            if polarity <= 0.56:
                description = "negative"
            elif 0.56 < polarity <= 0.67:
                description = "neutral"
            elif polarity > 0.67:
                description = "positive"
        else:
            if polarity < 0:
                description = "negative"
            elif polarity == 0:
                description = "neutral"
            elif polarity > 0:
                description = "positive"

        plt.title(
            "Overall {} emotion of {} Tweets on keyword {}.".format(
                description, str(num), keyword
            )
        )
        plt.ylabel("Percentage")
        plt.xlabel("Emotion")

        # Set tick colors:
        ax = plt.gca()
        ax.tick_params(axis="x", colors="blue")
        ax.tick_params(axis="y", colors="red")

        # Plot the data:
        my_colors = [
            "lightsalmon",
            "beige",
            "blue",
        ]

        s.plot(
            kind="bar", color=my_colors,
        )

        plt.show()

    def download_data(self, training=True):
        """

        Download all the tweets text in a csv file

        """
        if training:
            df = pd.DataFrame(
                {
                    "clean_text": self.text,
                    "sentiment": self.sentiment_,
                }
            )
        else:
            df = pd.DataFrame(
                {
                    "clean_text": self.text,
                    "sentiment": self.sentiment_vader,
                }
            )

        df.to_csv(self.path)

    def show_report(self, *args, training=True):
        """print the result of sentiment analysis by Training

        Parameters
        ----------
        *args : tuple of strings
            The percentage of different groups in final result
        training: Boolean
            True would return the report of the training model analysis,
            False would return the report of the vader method

        Returns
        -------
        Print the final report using Training
        """

        (polarity, positive, negative, neutral, key, n,) = args

        print("The emotion of {} tweets on {}. ".format(str(n), key))
        print()
        print("General Report: ")

        polarity = float(polarity)
        if training:
            if polarity <= 0.56:
                print("negative")
            elif 0.56 < polarity <= 0.67:
                print("neutral")
            elif polarity > 0.67:
                print("positive")
        else:
            if polarity < 0:
                print("negative")
            elif polarity == 0:
                print("neutral")
            elif polarity > 0:
                print("positive")

        print()
        print("Detailed Report: ")
        print("{} % people thought it was positive".format(str(positive)))
        print("{} % people thought it was negative".format(str(negative)))
        print("{} % people thought it was neutral".format(str(neutral)))

    @classmethod
    def input_user(cls):
        """

        setting your own Twitter API

        """

        return cls(
            consumer_key=input("Enter consumer_key: "),
            consumer_secret=input("Enter consumer_secret: "),
            access_token=input("Enter access_token: "),
            access_token_secret=input("Enter access_token_secret: "),
        )
