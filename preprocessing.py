import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Get the list of happy and sad emoticons
happy_emo = {
    "8)",
    "8-D",
    "8D",
    ":')",
    ":'-)",
    ":)",
    ":*",
    ":-)",
    ":-))",
    ":-D",
    ":-P",
    ":-b",
    ":-p",
    ":3",
    ":>",
    ":D",
    ":P",
    ":]",
    ":^)",
    ":^*",
    ":b",
    ":c)",
    ":d",
    ":o)",
    ":p",
    ":}",
    ";)",
    ";-)",
    ";-D",
    ";D",
    ";P",
    ";d",
    ";p",
    "<3",
    "=)",
    "=-3",
    "=-D",
    "=3",
    "=D",
    "=]",
    "=p",
    ">:)",
    ">:-)",
    ">:P",
    ">;)",
    "X-D",
    "X-P",
    "XD",
    "XP",
    "x)",
    "x-D",
    "x-p",
    "xD",
    "xd",
    "xp",
}

sad_emo = {
    ":'(",
    ":'-(",
    ":(",
    ":(>:(",
    ":-(",
    ":-/",
    ":-<",
    ":-[",
    ":-c",
    ":-||",
    ":/",
    ":<",
    ":@",
    ":L",
    ":S",
    ":[",
    ":\\",
    ":c",
    ":{",
    ":|",
    ";(",
    "=/",
    "=L",
    "=\\",
    ">.<",
    ">:/",
    ">:[",
    ">:\\",
}


def clean_text(text):
    """
    Clean text fro further analysis

    Parameters
    ----------
    text : str
        Original text

    Returns
    -------
    str
        Clean text without stopwords, html, @, RT, and translate happy and sad emoticons into happy and sad
    """

    new_text = []

    # Build the stopwords and remove 'no' and 'not' from the nltk stopwords
    stop_en = list(stopwords.words("english"))
    stop_en.remove("no")
    stop_en.remove("not")

    # Get a instance of the Stemmer and Lemmatizer
    s = PorterStemmer()
    l = WordNetLemmatizer()

    # make the text in lowercase
    text = text.lower()
    text = text.split(" ")
    for i in range(len(text)):
        if text[i] in happy_emo:
            text[i] = "happy"
        elif text[i] in sad_emo:
            text[i] = "sad"
    text = " ".join(text)

    # remove unnecessary characters and unicode
    text = re.sub(r"@[a-zA-Z0-9_]*", " ", text)
    text = re.sub(r"rt", " ", text)
    text = re.sub(r"https?:\S+|http?:\S|[^A-Za-z0-9]+", " ", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    text_list = word_tokenize(text)
    for i in text_list:
        i = s.stem(i)
        i = l.lemmatize(i)
        if i not in string.punctuation and i not in stop_en:
            new_text.append(i)

    return " ".join(new_text)


def share(part, whole):
    """Return the share of the part of the whole

    Parameters
    ----------
    part : int
        The number of the part
    whole : int
        whole number

    Returns
    -------
    float
        final percentage
    """
    return round((100 * part) / whole, 2)
