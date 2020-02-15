import pytest
from preprocessing import clean_text, share
from sentiment_analysis import sentiment_analysis
from tweets_download import TweetAnalysis
from nose.tools import assert_true


class TestClass:
    def test_preprocessing_clean(self):
        original = """MPCS 51046 is #awesome {}, and I learned a lot! :)
        This is a website I do know: https://twitter.com/home""".format(
            u"\U0001F600"
        )
        clean = "mpc awesom learn lot thi websit know"
        assert clean_text(original) == clean

    def test__preprocessing_share(self):
        part = 1
        whole = 3
        per_ = 33.33
        assert share(part, whole) == per_

    def test_sentiment_analysis_positive(self):
        positive_text = "MPCS 51046 is an awesome course and I love python!"
        assert sentiment_analysis(positive_text) > 0.8

    def test_sentiment_analysis_negative(self):
        negative_text = "I am so sad that this is the last class... :("
        assert sentiment_analysis(negative_text) < 0.2

    def test_sentiment_analysis_neutral(self):
        neutral_text = "This is a story of python."
        assert 0.57 < sentiment_analysis(neutral_text) < 0.67
        
    def test_tweets_download(self):
        positive_text = "MPCS 51046 is an awesome course and I love python!"
        negative_text = "I am so sad that this is the last class... :("
        neutral_text = "This is a story of python."
        ta = TweetAnalysis()
        ta.text = [positive_text, negative_text, neutral_text]
        ta.sentiment_ = []
        ta.sentiment_vader = []
        ta.num = 3
        ta.keyword = "is"
        result_training = ta.sentiment_training()
        result_vader = ta.get_sentiment_nltk()
        assert result_training == (0.51, 33.33, 33.33, 33.33, "is", 3)
        assert result_vader == (0.039900000000000005, 33.33, 33.33, 33.33, "is", 3)
        assert [
            round(ta.sentiment_[0] * 100, 2),
            round(ta.sentiment_[1] * 100, 2),
            round(ta.sentiment_[2] * 100, 2),
        ] == [91.07, 0.00, 61.93]
        assert [
            round(ta.sentiment_vader[0] * 100, 2),
            round(ta.sentiment_vader[1] * 100, 2),
            round(ta.sentiment_vader[2] * 100, 2),
        ] == [86.22, -74.25, 0.00]
