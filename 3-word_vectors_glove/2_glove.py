from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("glove.6B.100d.word2vec")

result = model.most_similar(positive=["woman", "king"], negative=["man"], topn=1)  # King - man + woman
print(result)

result = model.most_similar(positive=["ankara", "germany"], negative=["berlin"], topn=1)
print(result)

result = model.most_similar(positive=["teach", "doctor"], negative=["treat"], topn=1)
print(result)