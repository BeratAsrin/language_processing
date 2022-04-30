import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, CuDNNGRU
from tensorflow.python.keras.optimizers import adam_v2

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv("hepsiburada.csv")

target = dataset["Rating"].values.tolist()
data = dataset["Review"].values.tolist()

cutoff = int(len(data) * 0.80)  # 80% of data will be used for training and 20% for testing
x_train, x_test = data[:cutoff], data[cutoff:]
y_train, y_test = target[:cutoff], target[cutoff:]

num_words = 10000  # number of words to be contained in set
tokenizer = Tokenizer()  # A tokenizer object to be used in tokenizing process
tokenizer.num_words = num_words

tokenizer.fit_on_texts(data)  # Each word is tokenized by converting them into integers
# print(tokenizer.word_index)

x_train_tokens = tokenizer.texts_to_sequences(x_train)  # Each sentence is represented with integers
x_test_tokens = tokenizer.texts_to_sequences(x_test)  # Each sentence is represented with integers

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)

# If number of tokens in the sentence is less than max_tokens than the remaining part will be filled with 0s, contrary
# if number of tokens is greater than max_tokens than some tokens in the sentence will be removed.
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)
# print(x_train_pad.shape)

"""
idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))  # reverse the dict


def tokens_to_string(tokens):  # function to be used convert tokenized sentence with words
    words = [inverse_map[token] for token in tokens if token != 0]
    text = " ".join(words)
    return text
"""

model = Sequential()
embedding_size = 50  # Each word is represented with word vector with size of 50.

model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name="embedding_layer"))  # num_words words are represented with embedding_size sized word vectors

model.add(CuDNNGRU(units=16, return_sequences=True))
model.add(CuDNNGRU(units=8, return_sequences=True))
model.add(CuDNNGRU(units=4, return_sequences=False))

model.add(Dense(1, activation="sigmoid"))

optimizer = adam_v2.Adam(learning_rate=1e-3)

model.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

print(model.summary())

# train the model
model.fit(np.asarray(x_train_pad), np.asarray(y_train), epochs=5, batch_size=256)

# test the model using test set
test_result = model.evaluate(np.asarray(x_test_pad), np.asarray(y_test))
print("Accuracy: " + str(test_result[1]))

# test the model by giving another example
test_text = ["bu çok kötü bir ürün"]
test_text_tokens = tokenizer.texts_to_sequences(test_text)  # Each sentence is represented with integers
test_text_pad = pad_sequences(test_text_tokens, maxlen=max_tokens)
test_result = model.predict(x=test_text_pad)
print(test_result)
if test_result >= 0.5:
    print("Positive")
else:
    print("Negative")
