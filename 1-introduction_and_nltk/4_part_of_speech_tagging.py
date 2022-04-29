import nltk

text = "My name is Berat Asrin CAFEROGLU. I am currently working at Anadolu ISUZU."
word_tokens = nltk.word_tokenize(text)

print(nltk.pos_tag(word_tokens))

