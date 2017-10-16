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
stemmer = SnowballStemmer("english")
stWords = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

#####################################################

def arreglar(texto):
    lista_texto = tokenizer.tokenize(texto.lower())
    #Quitando Stopwords en ingles
    words_noStop = [wrd for wrd in lista_texto if not wrd in stWords]
    #Transformacion Snowball Stemmer
    stems = [stemmer.stem(t) for t in words_noStop]
    return(stems)

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
def leerDocumento(file):
    with open(file) as f:
        raw= f.read()
        texto_arreglado = arreglar(raw)
    return texto_arreglado


#Retorna el set de palabras conjuntas
def crearSetPalabras(lista_documentos):
    return set(word for words in lista_documentos for word in words)

#Retorna vectores de frecuencias de palabras de cada documento en columnas
#Se toman frecuencias respecto a set de palabras totales
def crearVectoresPalabras(set_palabras, lista_documentos):

    def vectorize(frequency_dict):
        return [frequency_dict.get(word, 0) for word in set_palabras]

    frequencies = map(contPalabras, lista_documentos)

    return list(map(vectorize, frequencies))

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
            simil_vectores = lambda centro: similaridad(centro, vector)
            centro = max(self.centros, key=simil_vectores)
            return self.centros.index(centro)

        self.clusters = [[] for c in self.centros]
        for vector in self.vectores:
            index = centroMasCercano(vector)
            self.clusters[index].append(vector)

    #Mueve nodos del centro a promedio de cada nodo de palabra del cluster
    def moverCentros(self):
        nuev_centros = []
        for cluster in self.clusters:
            centro = [average(ci) for ci in zip(*cluster)]
            nuev_centros.append(centro)
        if nuev_centros == self.centros:
            return False
        self.centros = nuev_centros
        return True

    def iterador(self):
        self.relacionar()
        while self.moverCentros():
            self.relacionar()

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


############################################################################

def main():
    start_time = time.time()

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    direccion = "Docs/*.txt"

    # División de los archivos entre procesadores
    if rank == 0:
        files = glob.glob(direccion)
        scatter_files = [[] for c in range(size)]

        for p in range(len(files)):
            i = p%size
            scatter_files[i].append(files[p])

    else:
        files = None
        scatter_files = None

    #realizar scatter
    archivos_tarea = comm.scatter(scatter_files, root=0)

    lista_docs_local = []
    lista_titulos_local = []

    #computar archivos locales
    for file in archivos_tarea:
        with open(file) as f:
            raw = f.read()
            texto_arreglado = arreglar(raw)
            lista_docs_local.append(texto_arreglado)
            lista_titulos_local.append(os.path.basename(f.name))

    archivos_tarea = [lista_docs_local,lista_titulos_local]

    #recibir resultado de todas las computaciones
    resultado = comm.gather(archivos_tarea, root=0)


    #Solo el main realiza el proceso de k-means
    if rank == 0:

        lista_docs_limpios = []
        lista_titulos = []

        for r in resultado:

            for doc in r[0]:
                lista_docs_limpios.append(doc)

            for titl in r[1]:
                lista_titulos.append(titl)

        set_palabras = crearSetPalabras(lista_docs_limpios)
        vec_frecuencias = crearVectoresPalabras(set_palabras, lista_docs_limpios)

        dicc_textos = dict(zip(map(str, vec_frecuencias), lista_titulos))
        kmeans = KMeans(2, vec_frecuencias)
        kmeans.iterador()

        mostrarResultados(kmeans.clusters, dicc_textos)
        print("-------TIEMPO DE EJECUCION: %s SEGUNDOS -------" % (time.time() - start_time))
    else:
        resultado = []


if __name__ == "__main__":
    main()
