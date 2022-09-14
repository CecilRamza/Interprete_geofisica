from agrupamiento import *
from herramientas import *
import pickle

###
# Agrupamiento
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
# El siguiente es un ejemplo de agrupamiento por métodos aglomerativos
#

archivo="Magnetometría/Base_de_datos-filtrada/7F.csv"
columnas=[2,3,4,5,6,12,13,14,15,16,17,18,19,20,21,27,28,29,30,31,37,38,39,40,41,47,48,49,50,51,57,58,59,60,61]
reduccion=[[1,2,3,4],[6,7,8,9],[11,12,13,14],[16,17,18,19],[21,22,23,24],[26,27,28,29],[31,32,33,34]]
varianzas=[80,80,80,80,80,80,80]
redes=[]
escalas=[]
n_redes=5
salidas=[]
escala=[]
estandarizacion=[]
estandarizaciones=[]
pca=[]
pcas=[]
componente=[]
componentes=[]
clasificados=[]

with open("estadarizaciones.p","rb") as infile:
    estandarizaciones=pickle.load(infile)

with open("pcas.p","rb") as infile:
    pcas=pickle.load(infile)

with open("componentes.p","rb") as infile:
    componentes=pickle.load(infile)

with open("escalamiento.p","rb") as infile:
    escala=pickle.load(infile)

with open("centroides.p","rb") as infile:
    kmedias=pickle.load(infile)

clasificada=clasificacion(archivo,columnas)
minimo=min(clasificada.y)
clasificada.y=clasificada.y-minimo
minimo=min(clasificada.x)
clasificada.x=clasificada.x-minimo
clasificada.reduccion_pca(reduccion,estandarizaciones,pcas,componentes)
clasificada.escalamiento(escala)
clasificada.kmeans(kmedias)
clasificada.malla(radio=0.1)
clasificada.etiquetar(1)
grafica(clasificada.clasificados,[2,3],[4,5,6],nombres=["Gradiente vertical [nT/m]","Etiquetas"],titulos=["Malla","Centroides k-medias (3D isomap)"],clasificacion=clasificada.malla_etiquetas)

#
# El siguiente es un ejemplo de agrupamiento por som
#

for i in range(n_redes):
    salidas.append("som"+str(i)+".p")

for i in range(n_redes):
    with open(salidas[i],"rb") as infile:
        redes.append(pickle.load(infile))

for i in range(n_redes):
    escala.append("escalamiento"+str(i)+".p")

for i in range(n_redes):
    with open(escala[i],"rb") as infile:
        escalas.append(pickle.load(infile))

for i in range(n_redes):
    estandarizacion.append("estandarizacion"+str(i)+".p")

for i in range(n_redes):
    with open(estandarizacion[i],"rb") as infile:
        estandarizaciones.append(pickle.load(infile))

for i in range(n_redes):
    pca.append("pca"+str(i)+".p")

for i in range(n_redes):
    with open(pca[i],"rb") as infile:
        pcas.append(pickle.load(infile))

for i in range(n_redes):
    componente.append("componente"+str(i)+".p")

for i in range(n_redes):
    with open(componente[i],"rb") as infile:
        componentes.append(pickle.load(infile))

for i in range(n_redes):
    clasificada=clasificacion(archivo,columnas,escalamiento_min=[-256.0433,-256.0433,-256.0433,-256.0433,-256.0433,-130.7216,-130.7216,-130.7216,-130.7216,-130.7216,-333.2798,-333.2798,-333.2798,-333.2798,-333.2798,0.0426,0.0426,0.0426,0.0426,0.0426,0.0025,0.0025,0.0025,0.0025,0.0025,-1.57,-1.57,-1.57,-1.57,-1.57,0.0017,0.0017,0.0017,0.0017,0.0017],escalamiento_max=[235.7932,235.7932,235.7932,235.7932,235.7932,104.7824,104.7824,104.7824,104.7824,104.7824,413.7691,413.7691,413.7691,413.7691,413.7691,1192.3648,1192.3648,1192.3648,1192.3648,1192.3648,776.2437,776.2437,776.2437,776.2437,776.2437,1.57,1.57,1.57,1.57,1.57,29.2427,29.2427,29.2427,29.2427,29.2427])
    minimo=min(clasificada.y)
    clasificada.y=clasificada.y-minimo
    minimo=min(clasificada.x)
    clasificada.x=clasificada.x-minimo
    clasificada.reduccion_pca(reduccion,estandarizaciones[i],pcas[i],componentes[i])
    clasificada.escalamiento(escalas[i])
    clasificada.som(redes[i],topologia="rectangular")
    clasificada.malla(metodo="red_rectangular",cumulos=3)
    clasificada.etiquetar(2)
    grafica(clasificada.clasificados,[2,3],[4,5,6],nombres=["Reducción al polo","Etiquetas"],titulos=["Malla","SOM - sensor inferior"],clasificacion=clasificada.malla_etiquetas)
    clasificados.append(clasificada)

for i in range(clasificada.clasificados.shape[0]):
    suma_r=0
    suma_g=0
    suma_b=0
    for j in range(n_redes):
        suma_r=suma_r+clasificados[j].clasificados[i,4]
        suma_g=suma_g+clasificados[j].clasificados[i,5]
        suma_b=suma_b+clasificados[j].clasificados[i,6]
    clasificada.clasificados[i,4]=round(suma_r/n_redes)
    clasificada.clasificados[i,5]=round(suma_g/n_redes)
    clasificada.clasificados[i,6]=round(suma_b/n_redes)

hsv=np.zeros((clasificada.clasificados.shape[0],3))

for i in range(clasificada.clasificados.shape[0]):
    r=clasificada.clasificados[i,4]/255
    g=clasificada.clasificados[i,5]/255
    b=clasificada.clasificados[i,6]/255
    pivote=np.array(colorsys.rgb_to_hsv(r,g,b))
    hsv[i,:]=pivote

histograma,cajones=np.histogram(hsv[:,1],bins="auto")
maximo=max(hsv[:,1])
for i in range(hsv.shape[0]):
    for j in range(cajones.shape[0]-1):
        if(hsv[i,1]>=cajones[j] and hsv[i,1]<=cajones[j+1]):
            break
    suma=0
    for k in range(j):
        suma=suma+(histograma[k]/hsv.shape[0])
    hsv[i,1]=suma

for i in range(clasificada.clasificados.shape[0]):
    rgb=np.array(colorsys.hsv_to_rgb(hsv[i,0],hsv[i,1],hsv[i,2]))
    clasificada.clasificados[i,4]=round(rgb[0]*255)
    clasificada.clasificados[i,5]=round(rgb[1]*255)
    clasificada.clasificados[i,6]=round(rgb[2]*255)

colores=np.zeros((clasificada.clasificados.shape[0],3))
colores[:,0]=clasificada.clasificados[:,4]
colores[:,1]=clasificada.clasificados[:,5]
colores[:,2]=clasificada.clasificados[:,6]
som=MiniSom(6,6,3,sigma=1,learning_rate=0.1,topology="rectangular")
som.random_weights_init(colores)
som.train(colores,colores.shape[0])
pesos=som.get_weights()

for i in range(colores.shape[0]):
    pivote=pesos[som.winner(colores[i])]
    clasificada.clasificados[i,4]=round(pivote[0])
    clasificada.clasificados[i,5]=round(pivote[1])
    clasificada.clasificados[i,6]=round(pivote[2])

mapa=np.reshape(pesos,(36,3))

for i in range(36):
    clasificada.malla_etiquetas[1][i,0]=round(mapa[i,0])
    clasificada.malla_etiquetas[1][i,1]=round(mapa[i,1])
    clasificada.malla_etiquetas[1][i,2]=round(mapa[i,2])

grafica(clasificada.clasificados,[2,3],[4,5,6],nombres=["Reducción al polo","Etiquetas"],titulos=["Malla","SOM promedio - sensor inferior"],clasificacion=clasificada.malla_etiquetas)
