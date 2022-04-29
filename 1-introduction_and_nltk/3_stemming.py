import nltk

porter_stemmer = nltk.stem.PorterStemmer()

words = ["drive", "driving", "driver", "drives", "drove", "cats", "children"]

stemmed_words = list()
for word in words:
    stemmed_words.append(porter_stemmer.stem(word))

print(words)
print(stemmed_words)
