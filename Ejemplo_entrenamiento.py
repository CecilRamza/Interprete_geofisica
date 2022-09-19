from entrenamiento import *
import pickle

###
# Entrenamiento
###

##
# "n_entrenar" es el número de archivos cuyos vectores se usarán para entrenar
# "centroides" es usado para el entrenamiento usando métodos de agrupamiento
# "bases" es una lista cuyos elementos son los nombres de archivos de los que se eligirán las bases de datos que se usarán para entrenar
# "columnas" es una lista con las columnas de las variables que se usarán para entrenar
# "reduccion" es una lista de listas que contiene los conjuntos de variables (columnas) a usar para reducir la dimensionalidad usando PCA, considerando únicamente las variables de la lista "columnas"
# "varianzas" es una lista cuyos elementos son la varianza que se usará para elegir las componentes principales del algoritmo PCA
##

#
# El siguiente es un ejemplo de entrenamiento por métodos aglomerativos
#

ciclo=True
redes=[]
salidas=[]
escalas=[]
escalamientos=[]
estandarizacion=[]
estandarizaciones=[]
pca=[]
pcas=[]
componente=[]
componentes=[]
red=0
n_redes=5
n_entrenar=5
centroides=3
bases=["2X.csv","3X.csv","4X.csv","5X.csv","6X.csv","2F.csv","3F.csv","4F.csv","5F.csv","6F.csv"]
columnas=[2,3,4,5,6,12,13,14,15,16,17,18,19,20,21,27,28,29,30,31,37,38,39,40,41,47,48,49,50,51,57,58,59,60,61]
reduccion=[[1,2,3,4],[6,7,8,9],[11,12,13,14],[16,17,18,19],[21,22,23,24],[26,27,28,29],[31,32,33,34]]
varianzas=[80,80,80,80,80,80,80]

indices=random.sample(bases,n_entrenar)
archivos=[]
for i in range(len(indices)):
    archivos.append("Magnetometría/Base_de_datos-filtrada/"+indices[i])

entrenada=entrenamiento(archivos,columnas,escalamiento_min=[-256.0433,-256.0433,-256.0433,-256.0433,-256.0433,-130.7216,-130.7216,-130.7216,-130.7216,-130.7216,-333.2798,-333.2798,-333.2798,-333.2798,-333.2798,0.0426,0.0426,0.0426,0.0426,0.0426,0.0025,0.0025,0.0025,0.0025,0.0025,-1.57,-1.57,-1.57,-1.57,-1.57,0.0017,0.0017,0.0017,0.0017,0.0017],escalamiento_max=[235.7932,235.7932,235.7932,235.7932,235.7932,104.7824,104.7824,104.7824,104.7824,104.7824,413.7691,413.7691,413.7691,413.7691,413.7691,1192.3648,1192.3648,1192.3648,1192.3648,1192.3648,776.2437,776.2437,776.2437,776.2437,776.2437,1.57,1.57,1.57,1.57,1.57,29.2427,29.2427,29.2427,29.2427,29.2427])
entrenada.reduccion_pca(reduccion,varianzas)
entrenada.escalamiento(metodo="minmax")
entrenada.kmeans(centroides)

etiquetas=entrenada.kmedias.predict(entrenada.conj_entr)
pivote=np.ones(etiquetas.shape[0])
pivote=np.array(pivote,dtype=bool)
histograma,cajones=np.histogram(etiquetas,bins=centroides)
maximo=np.argsort(histograma)[-1]
for i in range(etiquetas.shape[0]):
    if(etiquetas[i]==maximo):
        pivote[i]=False
entrenada.conj_entr=entrenada.conj_entr[pivote,:]
entrenada.kmeans(centroides)

with open("escalamiento.p","wb") as outfile:
    pickle.dump(entrenada.escala,outfile)

with open("estandarizaciones.p","wb") as outfile:
    pickle.dump(entrenada.estandarizaciones,outfile)

with open("pcas.p","wb") as outfile:
    pickle.dump(entrenada.pcas,outfile)

with open("componentes.p","wb") as outfile:
    pickle.dump(entrenada.componentes,outfile)

with open("centroides.p","wb") as outfile:
    pickle.dump(entrenada.kmeans,outfile)

#
# El siguiente es un ejemplo de entrenamiento por SOM
#

while ciclo:
    indices=random.sample(bases,n_entrenar)
    archivos=[]
    for i in range(len(indices)):
        archivos.append("Magnetometría/Base_de_datos-filtrada/"+indices[i])
    entrenada=entrenamiento(archivos,columnas,escalamiento_min=[-256.0433,-256.0433,-256.0433,-256.0433,-256.0433,-130.7216,-130.7216,-130.7216,-130.7216,-130.7216,-333.2798,-333.2798,-333.2798,-333.2798,-333.2798,0.0426,0.0426,0.0426,0.0426,0.0426,0.0025,0.0025,0.0025,0.0025,0.0025,-1.57,-1.57,-1.57,-1.57,-1.57,0.0017,0.0017,0.0017,0.0017,0.0017],escalamiento_max=[235.7932,235.7932,235.7932,235.7932,235.7932,104.7824,104.7824,104.7824,104.7824,104.7824,413.7691,413.7691,413.7691,413.7691,413.7691,1192.3648,1192.3648,1192.3648,1192.3648,1192.3648,776.2437,776.2437,776.2437,776.2437,776.2437,1.57,1.57,1.57,1.57,1.57,29.2427,29.2427,29.2427,29.2427,29.2427])
    entrenada.reduccion_pca(reduccion,varianzas)
    entrenada.escalamiento(metodo="minmax")
    entrenada.som(6,6,porcentaje_base=0.15,vecindad=0.5,aprendizaje=0.1,topologia="hexagonal")
    red=red+1
    redes.append(entrenada.mao)
    escalas.append(entrenada.escala)
    estandarizacion.append(entrenada.estandarizaciones)
    pca.append(entrenada.pcas)
    componente.append(entrenada.componentes)
    if(red==n_redes):
        ciclo=False

for i in range(n_redes):
    salidas.append("som"+str(i)+".p")
    escalamientos.append("escalamiento"+str(i)+".p")
    estandarizaciones.append("estandarizacion"+str(i)+".p")
    pcas.append("pca"+str(i)+".p")
    componentes.append("componente"+str(i)+".p")

for i in range(n_redes):
    with open(salidas[i],"wb") as outfile:
        pickle.dump(redes[i],outfile)
    with open(escalamientos[i],"wb") as outfile:
        pickle.dump(escalas[i],outfile)
    with open(estandarizaciones[i],"wb") as outfile:
        pickle.dump(estandarizacion[i],outfile)
    with open(pcas[i],"wb") as outfile:
        pickle.dump(pca[i],outfile)
    with open(componentes[i],"wb") as outfile:
        pickle.dump(componente[i],outfile)
