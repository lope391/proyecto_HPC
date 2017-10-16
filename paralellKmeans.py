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
import sys
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

# class KMeans(object):
#
#     #Constructor del objeto K-means
#     def __init__(self, k, vectores):
#         self.centros = random.sample(vectores,k)
#         self.clusters = [[] for c in self.centros]
#         self.vectores = vectores
#
#     #Relaciona cada documento con el centro mas cercano
#     def relacionar(self):
#         #Saca similiaridad entre cada centro y documentos. Devuelve el mas cercano
#         def centroMasCercano(vector):
#             simil_vectores = lambda centro: similaridad(centro, vector)
#             centro = max(self.centros, key=simil_vectores)
#             return self.centros.index(centro)
#
#         self.clusters = [[] for c in self.centros]
#         for vector in self.vectores:
#             index = centroMasCercano(vector)
#             self.clusters[index].append(vector)
#
#     #Mueve nodos del centro a promedio de cada nodo de palabra del cluster
#     def moverCentros(self):
#         nuev_centros = []
#         for cluster in self.clusters:
#             centro = [average(ci) for ci in zip(*cluster)]
#             nuev_centros.append(centro)
#         if nuev_centros == self.centros:
#             return False
#         self.centros = nuev_centros
#         return True
#
#     def iterador(self):
#         self.relacionar()
#         while self.moverCentros():
#             self.relacionar()

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


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')



############################################################################
def main():

    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object

    #El rank 0 esta dedicado a enviar tareas y recibir resultados
    #El rank 0 es el unico que realiza en k-means
    if rank == 0:

        start_time = time.time()
        direccion = "Gutenberg/*.txt"
        lista_titulos = []
        lista_documentos_limpios = []

        #Recoleccion de todos los documentos a evaluar
        #Asignación de numero de trabajadores
        files = glob.glob(direccion)
        tareas = range(len(files))
        ind_tarea = 0
        num_trabajadores = size - 1
        trab_cerrados = 0

        #Inicio del Task Pull
        while trab_cerrados < num_trabajadores:

            #Master recibe el estado actual de los workers
            datos = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            fuente = status.Get_source()
            tag = status.Get_tag()

            #Si en worker envia READY en el tag se le envía un documeto para limpiar
            #Si no hay mas documentos se envía una señal de terminación
            if tag == tags.READY:
                if ind_tarea < len(tareas):
                    comm.send(files[ind_tarea], dest=fuente, tag=tags.START)
                    ind_tarea += 1
                else:
                    comm.send(None, dest=fuente, tag=tags.EXIT)

            #Si recibe el tag DONE extrae los resultados y los guarda localmente
            elif tag == tags.DONE:
                resultados = datos
                titl = resultados[1]
                lista_titulos.append(os.path.basename(titl))
                lista_documentos_limpios.append(resultados[0])

            #Cuando se vuelve a recibir la señal de terminación cierra el worker
            elif tag == tags.EXIT:
                trab_cerrados += 1

        set_palabras = crearSetPalabras(lista_documentos_limpios)
        vec_frecuencias = crearVectoresPalabras(set_palabras,lista_documentos_limpios)
        dicc_textos = dict(zip(map(str, vec_frecuencias),lista_titulos))
        comm.Barrier()

    #Todos los otros nodos esperan una documento o una señal de terminación
    else:

        while True:
            comm.send(None, dest=0, tag=tags.READY)
            tarea = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                lista = leerDocumento(tarea)
                result = [lista,tarea]
                comm.send(result, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)
        comm.Barrier()


    def relacionar(vectores,centros):
        if rank == 0:

                tareas = range(len(vectores))
                ind_tarea = 0
                num_trabajadores = size - 1
                trab_cerrados = 0
                while trab_cerrados < num_trabajadores:
                    #Master recibe el estado actual de los workers
                    datos = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    fuente = status.Get_source()
                    tag = status.Get_tag()

                    #Si en worker envia READY en el tag se le envía un documeto para limpiar
                    #Si no hay mas documentos se envía una señal de terminación
                    if tag == tags.READY:
                        if ind_tarea < len(tareas):
                            comm.send([vectores[ind_tarea],centros], dest=fuente, tag=tags.START)
                            ind_tarea += 1
                        else:
                            comm.send(None, dest=fuente, tag=tags.EXIT)

                    #Si recibe el tag DONE extrae los resultados y los guarda localmente
                    elif tag == tags.DONE:
                        vector = datos[1]
                        index = datos[0]
                        clusters[index].append(vector)

                    #Cuando se vuelve a recibir la señal de terminación cierra el worker
                    elif tag == tags.EXIT:
                        trab_cerrados += 1

                comm.Barrier()
        #Todos los otros nodos esperan una documento o una señal de terminación

        else:

            while True:
                comm.send(None, dest=0, tag=tags.READY)
                tarea = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                if tag == tags.START:
                    vector = tarea[0]
                    centros = tarea[1]
                    simil_vectores = lambda centro: similaridad(centro, vector)
                    centro = max(centros, key=simil_vectores)
                    indice = centros.index(centro)
                    comm.send([indice,vector], dest=0, tag=tags.DONE)
                elif tag == tags.EXIT:
                    break

            comm.send(None, dest=0, tag=tags.EXIT)
            comm.Barrier()

            ##############################

    if rank == 0:
        #Mueve nodos del centro a promedio de cada nodo de palabra del cluster
        def moverCentros(centers):
            result = []
            nuev_centros = []
            for cluster in clusters:
                centro = [average(ci) for ci in zip(*cluster)]
                nuev_centros.append(centro)
            if nuev_centros == centers:
                result.append(False)
                result.append(centers)
            else:
                centros = nuev_centros
                result.append(True)
                result.append(centros)
            return result



        centros = random.sample(vec_frecuencias,2)
        clusters = [[] for c in centros]
        relacionar(vec_frecuencias,centros)
        resultado = moverCentros(centros)
        centros = resultado[1]
        while resultado[0]:
            clusters = [[] for c in centros]
            relacionar(vec_frecuencias,centros)
            resultado = moverCentros(centros)
            centros = resultado[1]



        for i in range(1,size):
            comm.send(None, dest=i, tag=tags.EXIT)

        mostrarResultados(clusters, dicc_textos)
        print("-------TIEMPO DE EJECUCION: %s SEGUNDOS -------" % (time.time()-start_time))

    else:
        while True:
            relacionar([],[])








if __name__ == "__main__":
    main()
