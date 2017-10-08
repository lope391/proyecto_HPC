import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

#nltk.download('punkt')
#nltk.download('stopwords')


stemmer = SnowballStemmer("english")
stWords = set(stopwords.words('english'))
print(stWords)

file = open("Gutenberg/ShortStory.txt")
raw = file.read()

raw = raw.lower()
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(raw)
#words = word_tokenize(raw)

print("Palabras originales:")
print(words)

print("---------------")
print("Quitando Stopwords en ingles:")
words_noStop = [wrd for wrd in words if not wrd in stWords]
print(words_noStop)

print("---------------")
print("Transformación Snowball Stemmer:")

stems = [stemmer.stem(t) for t in words_noStop]
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

