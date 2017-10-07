import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

stemmer = SnowballStemmer("english")

file = open("Gutenberg/ShortStory.txt")
raw = file.read()

words = word_tokenize(raw)

print("Palabras originales:")
print(words)

print("---------------")
print("Transformación Snowball Stemmer:")

stems = [stemmer.stem(t) for t in words]
print(stems)
print(len(stems))

print("---------------")
print("Set de palabras:")
stemSet = set(stems)
print(stemSet)
print(len(stemSet))

print("---------------")
print("Set de palabras con Pesos:")
frecuencias = nltk.FreqDist(stems)
print(frecuencias.tabulate())
