import nltk

text = "My name is Berat Asrin CAFEROGLU. I am currently working at Anadolu ISUZU."
tokens = nltk.tokenize.word_tokenize(text)

stopwords = nltk.corpus.stopwords.words("english")

filtered_tokens = list()
for token in tokens:
    if token not in stopwords:
        filtered_tokens.append(token)

print(tokens)
print(filtered_tokens)
