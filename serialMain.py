import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

#nltk.download('punkt')
#nltk.download('stopwords')


stemmer = SnowballStemmer("english")
stWords = set(stopwords.words('english'))
print(stWords)

def leerTexto(nombre):
    file = open(nombre)
    raw = file.read()
    raw = raw.lower()
    file.close()
    return raw

def arreglar(palabras):
    print("---------------")
    print("Quitando Stopwords en ingles:")
    words_noStop = [wrd for wrd in palabras if not wrd in stWords]
    print(words_noStop)
    print("---------------")
    print("Transformación Snowball Stemmer:")
    stems = [stemmer.stem(t) for t in words_noStop]
    print(stems)
    return(stems)

def palabrasTotales(lista1, lista2):
    all_words = list(set(lista1).union(set(lista2)))
    return all_words

texto1 = leerTexto("Gutenberg/ShortStory.txt")
texto2 = leerTexto("Gutenberg/ShortStory2.txt")

tokenizer = RegexpTokenizer(r'\w+')
words1 = tokenizer.tokenize(texto1)
words2 = tokenizer.tokenize(texto2)

words1 = arreglar(words1)
words2 = arreglar(words2)
#words = word_tokenize(raw)

print("Palabras originales de la lista 1:")
print(words1)
print("Palabras originales de la lista 2:")
print(words2)

print("Set de palabras totales")
print(palabrasTotales(words1,words2))
