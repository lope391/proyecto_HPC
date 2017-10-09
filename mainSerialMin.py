import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np
import random
import time
import glob
import os
from mpi4py import MPI
#nltk.download('punkt')
#nltk.download('stopwords')

start_time = time.time()
stemmer = SnowballStemmer("english")
stWords = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

#####################################################

def arreglar(texto):
    return([stemmer.stem(t) for t in [wrd for wrd in tokenizer.tokenize(texto.lower()) if not wrd in stWords]])

def contPalabras(palabras):
    contador = defaultdict(float)
    for pal in palabras:
        contador[pal] += 1.0 / len(palabras)
    return dict(contador)

def similaridad(vector1, vector2):
    return np.dot(vector1,vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

##################################################################

#Lee archivos de una carpeta
#Retorna matriz con palabras de cada documento por columna
def leerCarpeta(direccion):
    lista_titulos = []
    lista_documentos_limpios = []
    for file in glob.glob(direccion):
        with open(file) as f:
            lista_documentos_limpios.append(arreglar(f.read()))
            lista_titulos.append(os.path.basename(f.name))
    return lista_documentos_limpios, lista_titulos

#Retorna el set de palabras conjuntas
def crearSetPalabras(lista_documentos):
    return set(word for words in lista_documentos for word in words)

#Retorna vectores de frecuencias de palabras de cada documento en columnas
#Se toman frecuencias respecto a set de palabras totales
def crearVectoresPalabras(set_palabras, lista_documentos):

    def vectorize(frequency_dict):
        return [frequency_dict.get(word, 0) for word in set_palabras]
    return list(map(vectorize, map(contPalabras, lista_documentos)))

############################################################################
#Seccion de clustering por K-means

class KMeans(object):
    #Constructor del objeto K-means
    def __init__(self, k, vectores):
        self.centros = random.sample(vectores,k)
        self.clusters = [[] for c in self.centros]
        self.vectores = vectores
    #Relaciona cada documento con el centro mas cercano
    def relacionar(self):
        #Saca similiaridad entre cada centro y documentos. Devuelve el mas cercano
        def centroMasCercano(vector):
            #simil_vectores =
            centro = max(self.centros, key=lambda centro: similaridad(centro, vector))
            return self.centros.index(centro)

        self.clusters = [[] for c in self.centros]
        for vector in self.vectores:
            self.clusters[centroMasCercano(vector)].append(vector)
    #Mueve nodos del centro a promedio de cada nodo de palabra del cluster
    def moverCentros(self):
        nuev_centros = []
        for cluster in self.clusters:
            nuev_centros.append([average(ci) for ci in zip(*cluster)])
        if nuev_centros == self.centros:
            return False
        self.centros = nuev_centros
        return True

    def iterador(self):
        self.relacionar()
        while self.moverCentros():
            self.relacionar

def average(sequence):
    return sum(sequence) / len(sequence)


def mostrarResultados(clusters, dicc_textos):
    def buscadorDicc(vector):
        return dicc_textos[str(vector)]

    def buscadorCluster(cluster):
        return map(buscadorDicc, cluster)

    resultados = map(buscadorCluster, clusters)
    cont = 1
    for vec in resultados:
        print("Cluster " + str(cont))
        for ele in vec:
            print(ele)
        cont += 1



lista_documentos, lista_titulos = leerCarpeta("Gutenberg/*.txt")
set_palabras = crearSetPalabras(lista_documentos)
vec_frecuencias = crearVectoresPalabras(set_palabras,lista_documentos)

dicc_textos = dict(zip(map(str, vec_frecuencias),lista_titulos))
kmeans = KMeans(2,vec_frecuencias)
kmeans.iterador()

mostrarResultados(kmeans.clusters, dicc_textos)
print("-------TIEMPO DE EJECUCION: %s SEGUNDOS -------" % (time.time()-start_time))



############################################################################
