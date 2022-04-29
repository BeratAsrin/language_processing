from gensim.models import Word2Vec

f = open("hurriyet.txt", "r", encoding="utf8")
text = f.read()

sentence_list = text.split('\n')

corpus = list()

for sentence in sentence_list:
    corpus.append(sentence.split())

# sg=1 skip-gram, default value algorithm is cbow
trained_model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=5, sg=1)
trained_model.save("word2vec_hurriyet_skip_gram.model")
