from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove_input = "glove.6B.100d.txt"
word2vec_output = "glove.6B.100d.word2vec"
glove2word2vec(glove_input, word2vec_output)

model = KeyedVectors.load_word2vec_format(word2vec_output, binary=False)

print(model.most_similar("plato"))