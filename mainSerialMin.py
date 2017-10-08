import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np
import random
import glob
#nltk.download('punkt')
#nltk.download('stopwords')


stemmer = SnowballStemmer("english")
stWords = set(stopwords.words('english'))
# print(stWords)
tokenizer = RegexpTokenizer(r'\w+')

def leerCarpeta(direccion):
    lista_documentos = []
    files = glob.glob(direccion)
    for file in files:
        with open(file) as f:
            raw= f.read()
            texto_arreglado = arreglar(raw)
            lista_documentos.append(texto_arreglado)
    return lista_documentos


def arreglar(texto):
    return([stemmer.stem(t) for t in [wrd for wrd in tokenizer.tokenize(texto.lower()) if not wrd in stWords]])


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
    listaConjunta = palabrasTotales(words1,words2)

    dicFrecPal1 = contPalabras(words1)
    dicFrecPal2 = contPalabras(words2)

    vecFrecPal1 = [dicFrecPal1.get(word, 0) for word in listaConjunta]
    vecFrecPal2 = [dicFrecPal2.get(word, 0) for word in listaConjunta]

    return similaridad(vecFrecPal1,vecFrecPal2)


print(leerCarpeta("Gutenberg/*.txt"))


# texto1 = leerTexto("Gutenberg/John Ruskin___Modern Painters, Volume 4 (of 5).txt")
# texto2 = leerTexto("Gutenberg/Lucy Maud Montgomery___Lucy Maud Montgomery Short Stories, 1902 to 1903.txt")
#
# print(compararTextos(texto1,texto2))
#

############################################################################
