from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="NLP Sentiment Analysis on Twitter",
    version="1.0.0",
    description="NLP sentiment analysis on twitter based on keyword searching",
    long_description=long_description,
    url="https://github.com/mpcs-python/autumn-2019-project-hihowme",
    author="Haihao Guo",
    author_email="haihao@uchicago.edu",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="python twitter sentiment-analysis deep_learning nlp tweepy nltk",
    install_requires=[
        "tweepy",
        "numpy",
        "matplotlib",
        "nltk",
        "pandas",
        "gensim",
        "keras",
        "tensorflow",
        "scikit-learn",
    ],
)
