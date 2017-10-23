# Proyecto Integrador HPC
Proyecto Final Modulo HPC, Topicos Especiales en Telematica

Autores: Sarabia, Samuel. Carvajal, Lope


# Manual de Uso

## Funcionalidad
El programa está diseñado para tomar un grupo de documentos de texto que divide entre un numero de clusters definido dentro del codigo. 
El programa se ejecuta sobre la carpeta *Docs* y tomará todos los archivos.txt dentro de esta carpeta luego los dividirá en el numero de clusters definidos y para fines practicos imprime cada cluster con los documentos que contiene.

## Build
El proyecto esta hecho para ser ejecutado en un ambiente python 3. El manejo de paquetes es realizado por pip con la ayuda de un Makefile y el comando

    make pip
    
Este comando instalará con pip las dependencias necesarias del programa

## Parametros y Restricciónes
* El programa siempre corre sobre la carpeta Docs pero dentro de cualquier main existe la linea para editarlo.
  * El programa utiliza por default encoding UTF-8 que depués fue cambiado a latin 1 en un commit mas reciente .
* El numero de clusters que se buscará en el k-means esta definido dentro del main de cada archivo si se quiere cambiar.
  * El numero de clusters debe ser menor al numero de archivos existentes de lo contrario terminará con un error.
  
## Ejecución
El programa se ejecuta utilizando el executor.sh el cual está configurado por default a ejecutar el main parallelo en 4 cores con mpi. Para ejecutarlo se puede utilizar 
    
    sh executor.sh
    bash executor.sh
    
Se puede ejecutar un archivo especifico con el comando 

    mpiexec -np [numero de cores] python [nombre_archivo]
    
Existe un main serial que se corre con el comando 

    python serialMain.py
    
