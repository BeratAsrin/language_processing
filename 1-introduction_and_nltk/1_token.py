import nltk

text = "My name is Berat Asrin CAFEROGLU. I am currently working at Anadolu ISUZU."

word_tokens = nltk.tokenize.word_tokenize(text)  # tokenizes the words in the text
sent_tokens = nltk.tokenize.sent_tokenize(text)  # tokenizes the sentences in the text

print(word_tokens)
print(sent_tokens)