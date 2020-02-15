# sentiment analysis.py


import pickle
from gensim.models import Word2Vec
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


w2v_model = Word2Vec.load("training_result/model.w2v")

model = load_model("training_result/model.h5")

handle = open("training_result/tokenizer.pkl", "rb")
t = pickle.load(handle)
handle.close()

handle = open("training_result/encoder.pkl", "rb")
e = pickle.load(handle)
handle.close()


def sentiment_analysis(text):
    """
    Analyze the sentiment of given text
    Parameters
    ----------
    text : str
        Original text
    Returns
    -------
    float
        A float number representing the sentiment of the text. In particular, if the float number is less than 0.56,
        the text is negative, is the float number is bigger than 0.67, it is a positive text, otherwise the text is
        neutral.
    """

    return float(
        model.predict(pad_sequences(t.texts_to_sequences([text]), maxlen=100))[0]
    )
