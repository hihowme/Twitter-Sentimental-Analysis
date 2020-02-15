from tweets_download import TweetAnalysis

if __name__ == "__main__":

    # run this to input your own API
    # sa = TweetAnalysis.input_user()

    # initialize the class
    sa = TweetAnalysis(
        keyword=input("Enter Keyword: "), num=int(input("Enter Number: "))
    )
    sa.get_text()

    # nltk method
    analysis = sa.get_sentiment_nltk()
    sa.bar_plot(*analysis, training=False)
    sa.show_report(*analysis, training=False)

    # training method
    analysis_ = sa.sentiment_training()
    sa.bar_plot(*analysis_)
    sa.show_report(*analysis_)
    sa.download_data()
