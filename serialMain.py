import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np
import random
#nltk.download('punkt')
#nltk.download('stopwords')


stemmer = SnowballStemmer("english")
stWords = set(stopwords.words('english'))
print(stWords)
tokenizer = RegexpTokenizer(r'\w+')

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

def contPalabras(palabras):
    contador = defaultdict(float)
    for pal in palabras:
        contador[pal] += 1.0 / len(palabras)
    return dict(contador)

def similaridad(vector1, vector2):
    return np.dot(vector1,vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def compararTextos(texto1, texto2):
    words1 = tokenizer.tokenize(texto1)
    words2 = tokenizer.tokenize(texto2)
    words1 = arreglar(words1)
    words2 = arreglar(words2)
    listaConjunta = palabrasTotales(words1,words2)

    dicFrecPal1 = contPalabras(words1)
    dicFrecPal2 = contPalabras(words2)

    vecFrecPal1 = [dicFrecPal1.get(word, 0) for word in listaConjunta]
    vecFrecPal2 = [dicFrecPal2.get(word, 0) for word in listaConjunta]

    return similaridad(vecFrecPal1,vecFrecPal2)



texto1 = leerTexto("Gutenberg/ShortStory.txt")
texto2 = leerTexto("Gutenberg/ShortStory2.txt")

print(compararTextos(texto1,texto2))
