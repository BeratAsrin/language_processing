import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def closest_words_plot(model, word):
    word_vectors = np.empty((0, 100))
    word_labels = [word]

    close_words = model.wv.most_similar(word)

    word_vectors = np.append(word_vectors, np.array([model.wv[word]]), axis=0)

    for word, score in close_words:
        word_labels.append(word)
        word_vectors = np.append(word_vectors, np.array([model.wv[word]]), axis=0)

    tsne = TSNE(random_state=0)
    Y = tsne.fit_transform(word_vectors)

    x_cord = Y[:, 0]
    y_cord = Y[:, 1]

    plt.scatter(x_cord, y_cord)

    for label, x, y in zip(word_labels, x_cord, y_cord):
        plt.annotate(label, xy=(x, y), xytext=(5, -2), textcoords="offset points")

    plt.show()


loaded_model = Word2Vec.load("word2vec_hurriyet_skip_gram.model")
print(loaded_model.wv.most_similar("berlin"))
closest_words_plot(loaded_model, "berlin")
