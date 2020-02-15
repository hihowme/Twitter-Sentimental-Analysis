import pandas as pd
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from preprocessing import *
from sklearn.metrics import confusion_matrix, accuracy_score

# import training file
print("start this game!")
total_df = pd.read_csv("dataset/train.csv", encoding="ISO-8859-1")
total_df.columns = ["ItemID", "sentiment", "text"]
total_df.drop(["ItemID"], axis=1, inplace=True)
total_df.dropna(inplace=True)
total_df["text"] = total_df["text"].apply(clean_text)
total_df["sentiment"] = total_df["sentiment"].apply(float)
total_df.sentiment = total_df.sentiment.replace(1.0, "positive")
total_df.sentiment = total_df.sentiment.replace(0.0, "negative")


train_df, test_df = train_test_split(total_df, test_size=0.2, random_state=42)


train_text = train_df.text.tolist()
train_sentiment = train_df.sentiment.tolist()
test_text = test_df.text.tolist()
test_sentiment = test_df.sentiment.tolist()

print("finish csv!")
# Build W2V model and train
train_text_w2v = [i.split() for i in train_text]

w2v = Word2Vec(train_text_w2v, size=200, window=5, min_count=5, workers=8)
print(w2v.most_similar("New York"))

# save model
w2v.save("model.w2v")

# Pad_tokenize text
t = Tokenizer()
t.fit_on_texts(train_text)
n_v = len(t.word_index) + 1

# save model
pickle.dump(t, open("tokenizer.pkl", "wb"), protocol=0)


x_train = pad_sequences(t.texts_to_sequences(train_text), maxlen=100)
x_test = pad_sequences(t.texts_to_sequences(test_text), maxlen=100)

# Encoder
e = LabelEncoder()
e.fit(train_sentiment)

# save model
pickle.dump(e, open("encoder.pkl", "wb"), protocol=0)

y_train = e.transform(train_sentiment).reshape(-1, 1)
y_test = e.transform(test_sentiment).reshape(-1, 1)

# Embedding
embed_matrix = np.zeros((n_v, 200))
for i, j in t.word_index.items():
    if i in w2v.wv:
        embed_matrix[j] = w2v.wv[i]

embedding_layer = Embedding(
    n_v, 200, weights=[embed_matrix], input_length=100, trainable=False
)
print("finish embed!")


# Define the model
def create_model(layer):
    model = Sequential()
    model.add(layer)
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="relu"))
    model.compile(loss="mse", optimizer="adam", metrics=["mse"])
    return model


callbacks = [
    ReduceLROnPlateau(monitor="val_loss", patience=5, cooldown=0),
    EarlyStopping(monitor="val_acc", min_delta=1e-4, patience=5),
]

model = create_model(embedding_layer)

history = model.fit(
    x_train,
    y_train,
    batch_size=256,
    epochs=50,
    validation_split=0.1,
    verbose=1,
    callbacks=callbacks,
)

# save model
model.save("model.h5")

# Comparing the Training and validation loss
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(loss))

plt.plot(epochs, loss, "b", label="Training loss")
plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()

plt.show()

# We choose 0.57 as a threshold here based on several trying with the model
def get_sentiment(score):
    return "negative" if score < 0.57 else "positive"


y_pred_1d = []
y_test_1d = test_sentiment
scores = model.predict(x_test, verbose=1, batch_size=8000)
y_pred_1d = [get_sentiment(score) for score in scores]

# Plot confusion matrix
labels = ["negative", "positive"]
cm = confusion_matrix(y_test_1d, y_pred_1d, labels)
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
thresh = cm.max() / 2.0
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title("Confusion matrix", fontsize=20)
ax.set_xticklabels([""] + labels, fontsize=10)
ax.set_yticklabels([""] + labels, fontsize=10)
plt.imshow(cm, cmap=plt.cm.YlGn)
plt.colorbar()
plt.xlabel("Predicted", fontsize=20)
plt.ylabel("True", fontsize=20)
plt.rcParams["figure.figsize"] = (8, 8)
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j], ".2f"),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.show()


# Get the overall accuracy score
accuracy_score(y_test_1d, y_pred_1d)
