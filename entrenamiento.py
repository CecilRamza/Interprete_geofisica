import random
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from matplotlib import cm, colorbar
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from matplotlib.patches import RegularPolygon
from pyclustering.cluster.kmedians import kmedians
from mpl_toolkits.axes_grid1 import make_axes_locatable
from herramientas import *

########################################################
#                                                      #
# Esta clase prepara la base de datos a usar para el   #
# posterior entrenamiento, pudiendo ser k-means,       #
# k-medians y SOM.                                     #
#                                                      #
# entrenamiento(archivos,columnas), donde "archivos"   #
# es una lista que contiene las rutas a los archivos a #
# usar para entrenar, "columnas" es una lista con los  #
# números de columna que se usarán para entrenar.      #
#                                                      #
# entrenamiento.reduccion_pca(columnas,porcentajes),   #
# donde "columnas" es una lista de listas que contiene #
# las columnas a las que se desea aplicar reducción de #
# la dimensionalidad, "porcentajes" es una lista con   #
# los porcentajes de varianza mínimos deseados para    #
# elegir el número de componentes principales.         #
# entrenamiento.pcas es una lista con las componentes  #
# principales usadas para la reducción de              #
# dimensionalidad, útil para aplicar el mismo          #
# procedimiento al conjunto de validación.             #
# entrenamiento.estandarizaciones es una lista con los #
# parámetros de estandarización usados para cada       #
# conjunto de variables ingresado, útil para aplicar   #
# al conjunto de validación. entrenar.componentes es   #
# el número de componentes usadas para reducir cada    #
# conjunto de variables ingresado, útil para aplicar   #
# al conjunto de validación.                           #
#                                                      #
# entrenamiento.escalamiento(metodo), donde "metodo"   #
# el método a usar para escalar los valores de cada    #
# variable, las opciones son: "estandarizar", "minmax  #
# y "maxabs". entrenamiento.escala guarda el modelo    #
# para ser usado en el conjunto de validación.         #
#                                                      #
# entrenamiento.kmeans(num_cum), donde "num_cum" es el #
# número de centroides que se usarán para el           #
# entrenamiento. entrenamiento.kmedias guarda los      #
# centroides para aplicar al conjunto de validación.   #
#                                                      #
# entrenamiento.kmedians(num_cum), donde "num_cum" es  #
# el número de centroides que se usarán para el        #
# entrenamiento. entrenamiento.kmedianas guarda los    #
# centroides para aplicar al conjunto de validación.   #
#                                                      #
# entrenamiento.kmedoids(num_cum), donde "num_cum" es  #
# el número de centroides que se usarán para el        #
# entrenamiento. entrenamiento.kmedoides guarda los    #
# centroides para aplicar al conjunto de validación.   #
# Aún no es del todo aplicable, dado que la            #
# implementación requiere demasiada memoria.           #
#                                                      #
# entrenamiento.som(x,y,iteraciones,vecindad,aprendizaje,topologia), #
# donde "x" y "y" son las dimensiones de la red,       #
# "vecindad" es el parámetro de vecindad para la       #
# actualización de los pesos, "aprendizaje" es el      #
# factor de aprendizaje, "topologia" es la topología   #
# de la red, pudiendo ser "hexagonal" o "rectangular". #
# entrenamiento.mau almacena la red entrenada. A la    #
# salida genera el gráfico del mapa de distancias y    #
# el gráfico de errores topológico y de cuantización.  #
#                                                      #
########################################################

class entrenamiento:
    def __init__(self,archivos,columnas,escalamiento_min=[],escalamiento_max=[]):
        tamanos=[]
        for ijk in range(len(archivos)): #Ciclo que itera sobre el número de archivos
            d=open(archivos[ijk])
            datos=np.genfromtxt(d,delimiter=",") #Carga del archivo correspondiente
            if(ijk==0): #Si es el primer archivo, se inicializa el arreglo - tamaño de las variables
                n=datos.shape[1]
                self.conj_entr=np.zeros((0,n))
            tamanos.append(datos.shape[0])
            self.conj_entr=np.append(self.conj_entr,datos,axis=0) #Se almacena todo el contenido del archivo
            d.close()
        mascara=np.zeros(n,dtype=bool) #Inicialización de mascara booleana de tamaño igual al número de variables
        mascara[columnas]=True #Las columnas elegidas se marcan como verdaderas
        self.conj_entr=self.conj_entr[:,mascara] #Se filtran las variables elegidas
        self.nn=self.conj_entr.shape[0] #Se almacena el tamaño de la base de datos
        n=self.conj_entr.shape[1]
        if(len(escalamiento_min)>0 and len(escalamiento_max)>0 and len(escalamiento_min)==len(escalamiento_max)):
            inferior=0
            for k in range(len(tamanos)):
                superior=inferior+tamanos[k]
                for i in range(n):
                    pivote=self.conj_entr[inferior:superior,i]
                    rmin=min(pivote)
                    rmax=max(pivote)
                    for j in range(pivote.shape[0]):
                        pivote[j]=(pivote[j]-rmin)/(rmax-rmin)*(escalamiento_max[i]-escalamiento_min[i])+escalamiento_min[i]
                    self.conj_entr[inferior:superior,i]=pivote
                inferior=superior
    def reduccion_pca(self,columnas,porcentajes):
        sin_cambios=np.arange(self.conj_entr.shape[1]) #Arreglo de tamaño igual al número de columnas
        npo=len(porcentajes) #Número de reducciones deseadas
        for columna in columnas: #Ciclo que itera sobre el conjunto de columnas deseadas
            for digito in columna: #Ciclo que itera sobre cada columna del conjunto de columnas
                if digito in sin_cambios: #Si la columna se encuentra en el arreglo que contiene todas las columnas, se elimina dicha columna
                    sin_cambios=np.delete(sin_cambios,np.where(sin_cambios==digito)[0])
        conjunto=self.conj_entr[:,sin_cambios] #Se almacena la base de datos que no será reducida
        self.pcas=[] #Inicialización de una lista que contendrá las componentes principales de cada conjunto deseado
        self.estandarizaciones=[] #Inicialización de una lista que contendrá la estandarización aplicada a cada conjunto
        for columna in columnas: #Ciclo que itera sobre los conjuntos de columnas
            arreglo=self.conj_entr[:,columna] #Arreglo auxiliar que almacena cada conjunto a aplicar pca
            escala=preprocessing.StandardScaler().fit(arreglo) #Estandarización
            self.estandarizaciones.append(escala) #Se añade la estandarización anterior al arreglo que las almacena
            arreglo=escala.transform(arreglo) #Se aplica la estandarización
            pca=PCA() #Inicialización de pca
            self.pcas.append(pca.fit(arreglo)) #Se añaden los parámetros del pca actual al arreglo que los almacena
            self.componentes=np.zeros(npo) #Inicialización de arreglo para almacenar el número de componentes reducidas para cada conjunto ingresado
        for i in range(npo): #Ciclo que itera sobre el número de conjuntos solicitados
            varianzas=self.pcas[i].explained_variance_ratio_ #Arreglo que almacena las varianzas de cada componente para el conjunto actual
            suma=0 #Variable auxiliar para sumatoria de varianza
            for varianza in varianzas: #Ciclo que itera sobre la varianza de cada componente calculada
                self.componentes[i]=self.componentes[i]+1 #Se mide el número de componentes
                suma=suma+varianza #Sumatoria de la varianza
                if suma>=porcentajes[i]: #Si la suma de las varianzas es igual o mayor al umbral deseado se sale del ciclo
                    break
            pivote=self.conj_entr[:,columnas[i]]
            arreglo=np.matmul(pivote,np.transpose(self.pcas[i].components_[0:int(self.componentes[i]),:])) #Se aplica la reducción de pca considerando las componentes obtenidas del ciclo anterior
            arr=arreglo.reshape((self.nn,int(self.componentes[i]))) #Se cambian las dimensiones del arreglo anterior para concordar con la base de datos
            conjunto=np.append(conjunto,arr,axis=1) #Al conjunto que no recibe cambios se le agregan las columnas reducidas
        self.conj_entr=conjunto #Al final, el conjunto de entrenamiento se iguala al conjunto procesado
    def escalamiento(self,metodo="maxabs"):
        if(metodo=="estandarizar"): #Estandarización
            self.escala=preprocessing.StandardScaler().fit(self.conj_entr) #Escala
            self.conj_entr=self.escala.transform(self.conj_entr) #Aplicación
        elif(metodo=="minmax"): #Escalado entre -1 y 1
            self.escala=preprocessing.MinMaxScaler() #Escala
            self.conj_entr=self.escala.fit_transform(self.conj_entr) #Aplicación
        elif(metodo=="maxabs"): #Escalado entre 0 y 1
            self.escala=preprocessing.MaxAbsScaler() #Escala
            self.conj_entr=self.escala.fit_transform(self.conj_entr) #Aplicación
    def kmeans(self,num_cum):
        self.kmedias=KMeans(n_clusters=num_cum).fit(self.conj_entr) #Entrenamiento
    def kmedians(self,num_cum):
        m=self.conj_entr.shape[1]
        centroides=[]
        for i in range(num_cum):
            arreglo=np.random.rand(m)
            centroides.append(arreglo)
        self.kmedianas=kmedians(self.conj_entr,centroides) #Inicialización del algoritmo con el conjunto de entrenamiento y el número de cúmulos deseados
        self.kmedianas.process() #Entrenamiento
    def kmedoids(self,num_cum): #Aun no aplicable, pide mucha memoria
        self.kmedoides=KMedoids(n_clusters=num_cum,method="pam",init="k-medoids++").fit(self.conj_entr) #Entrenamiento
    def som(self,x,y,porcentaje_base=0.15,vecindad=1.,aprendizaje=1.,topologia="hexagonal"):
        m=self.conj_entr.shape[1] #Número de variables en el conjunto de entrenamiento
        rango=np.arange(self.nn)
        cuantizacion=[]
        topologico=[]
        eje=[]
        self.mao=MiniSom(x,y,m,sigma=vecindad,learning_rate=aprendizaje,topology=topologia) #Inicialización del algoritmo
#        self.mao.pca_weights_init(self.conj_entr)
        self.mao.random_weights_init(self.conj_entr) #Inicialización de los pesos de las neuronas
        elementos=random.sample(sorted(rango),int(self.nn*porcentaje_base))
        conjunto=self.conj_entr[elementos,:]
        print()
        if(topologia=="rectangular"):
            for j in range(conjunto.shape[0]):
                print("\r Iteración: "+str(j+1)+" de "+str(conjunto.shape[0]),end='')
                self.mao.update(conjunto[j],self.mao.winner(conjunto[j]),j,conjunto.shape[0])
                if(j%10000==0):
                    cuantizacion.append(self.mao.quantization_error(conjunto))
                    topologico.append(self.mao.topographic_error(conjunto)) #Cálculo del error topológico para redes rectangulares, se usa la implementada en la biblioteca minisom
                    eje.append(j)
            plt.pcolor(self.mao.distance_map(),cmap="autumn_r") #Graficación de la matriz de distancias
            plt.colorbar()
            plt.title("U-matrix")
            plt.savefig("SOM.png")
            plt.clf()
        elif(topologia=="hexagonal"):
            for j in range(conjunto.shape[0]):
                print("\r Iteración: "+str(j+1)+" de "+str(conjunto.shape[0]),end='')
                self.mao.update(conjunto[j],self.mao.winner(conjunto[j]),j,conjunto.shape[0])
                if(j%10000==0):
                    cuantizacion.append(self.mao.quantization_error(conjunto))
                    topologico.append(error_topologico_hexagonal(conjunto,self.mao)) #Cálculo del error tolológico para redes hexagonales, se usa implementación propia
                    eje.append(j)
            xx, yy=self.mao.get_euclidean_coordinates() #A partir de aquí se hace la graficación de la matriz de distancias, algoritmo tomado de la documentación de la biblioteca minisom
            umatrix=self.mao.distance_map()
            weights=self.mao.get_weights()
            f=plt.figure(figsize=(10,10))
            ax=f.add_subplot(111)
            ax.set_aspect('equal')
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    wy=yy[(i,j)]*np.sqrt(3)/2
                    hexa=RegularPolygon((xx[(i,j)],wy),numVertices=6,radius=.95/np.sqrt(3),facecolor=plt.cm.autumn_r(umatrix[i,j]),edgecolor='black')
                    ax.add_patch(hexa)
            xran=np.arange(weights.shape[0])
            yran=np.arange(weights.shape[1])
            plt.xticks(xran-.5,xran)
            plt.yticks(yran*np.sqrt(3)/2,yran)
            divider=make_axes_locatable(plt.gca())
            ax_cb=divider.new_horizontal(size="5%",pad=0.05)
            cb1=colorbar.ColorbarBase(ax_cb,cmap=cm.autumn_r,orientation='vertical')
            cb1.ax.get_yaxis().labelpad=int(7*7)
            cb1.ax.set_ylabel('Distancia de las neuronas con sus vecinos',rotation=270, fontsize=16)
            plt.gcf().add_axes(ax_cb)
            plt.savefig('SOM.png')
            plt.clf()
        print("\n")
        plt.plot(eje,cuantizacion,label="Error de cuantización") #Graficación de los errores de cuantización y topológico en el mismo gráfico
        plt.plot(eje,topologico,label="Error topológico")
        plt.grid(True)
        plt.grid(ls='-',lw=0.25)
        plt.ylabel("Error")
        plt.xlabel("Iteración")
        plt.title("Gráfico de errores, mapa auto-organizado")
        plt.legend()
        plt.savefig("SOM-errores.png")
        plt.clf()
