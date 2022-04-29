import nltk

lem = nltk.stem.WordNetLemmatizer()

words = ["drive", "driving", "driver", "drives", "drove", "cats", "children"]

for word in words:
    print(lem.lemmatize(word))

print(lem.lemmatize("driving", "v"))  # v for verb
print(lem.lemmatize("drove", "v"))  # v for verb
