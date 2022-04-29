import nltk

text = "Steve Jobs was an American entrepreneur, inventor, business magnate, media proprietor, and investor."

tokens = nltk.word_tokenize(text)
tagged_tokens = nltk.pos_tag(tokens)  # Part of speech
named_entities = nltk.ne_chunk(tagged_tokens)

named_entities.draw()