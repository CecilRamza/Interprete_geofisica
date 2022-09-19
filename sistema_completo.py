import vtk
import pywt
import math
import random
import colorsys
import subprocess
import numpy as np
import pyvista as pv
from resipy import Project
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import sklearn.manifold as manifold
from matplotlib import cm, colorbar
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from sklearn_extra.cluster import KMedoids
from matplotlib.colors import ListedColormap
from matplotlib.patches import RegularPolygon
from pyclustering.cluster.kmedians import kmedians
from sklearn.ensemble import IsolationForest as IsFo
from sklearn.neighbors import LocalOutlierFactor as LOF
from mpl_toolkits.axes_grid1 import make_axes_locatable

########################################################
#                                                      #
#       Cálculo de distancia, 2 para euclidiana        #
#    "x" y "y" son arreglos de una dimensión cuyos     #
#             tamaños deben ser iguales.               #
#                                                      #
########################################################

def metrica(x,y,grado):
    n=x.shape[0] #Tamaño del areglo
    distancia=0. #Inicialización
    for i in range(n):
        distancia=distancia+pow(x[i]-y[i],grado) #Suma parcial de los argumentos de la potencia
    distancia=pow(distancia,1./grado) #Cálculo de la potencia
    return distancia

def error_topologico_hexagonal(datos,som):
    n=datos.shape[0]
    coordenadas=som.get_euclidean_coordinates()
    error=0
    dist=math.sqrt(1.25)
    for i in range(n):
        ganadora=list(som.winner(datos[i,:]))
        mapa=som._activation_map
        for j in range(mapa.shape[0]):
            for k in range(mapa.shape[1]):
                cs=[j,k]
                if(ganadora!=cs):
                    break
            if(ganadora!=cs):
                break
        segunda=mapa[cs[0],cs[1]]
        for j in range(mapa.shape[0]):
            for k in range(mapa.shape[1]):
                c=[j,k]
                if(mapa[j,k]<segunda and c!=ganadora):
                    segunda=mapa[j,k]
                    cs=c
        distancia=metrica(np.array([coordenadas[0][ganadora[0]][ganadora[1]],coordenadas[1][ganadora[0]][ganadora[1]]]),np.array([coordenadas[0][cs[0]][cs[1]],coordenadas[1][cs[0]][cs[1]]]),2)
        if(distancia>dist):
            error=error+1
    return error/n

########################################################
#                                                      #
#     Ordenamiento de los "nodos" de acuerdo con la    #
#            cercanía que tengan con "nodo".           #
#    Regresa los índices ordenados desde el nodo más   #
#           cercano hasta el nodo más lejano.          #
#    "nodos" debe ser un arreglo de dimensiones [n,m], #
#       donde "n" es el número de puntos y "m" las     #
#  coordenadas de cada uno, "nodo" debe ser un arreglo #
#             de una dimensión de tamaño "m"           #
#                                                      #
########################################################

def cercanos(nodo,nodos):
    nod=np.asarray(nodos)
    deltas=nod-nodo
    dist=np.einsum('ij,ij->i',deltas,deltas)
    return np.argsort(dist)

########################################################
#                                                      #
#   Genera el gráfico de dispersión para una base de   #
#                         datos.                       #
#   "datos" debe ser un arreglo de dimensiones [n,m],  #
#  donde "n" es el número de datos y "m" es el número  #
#            de variables que lo definen.              #
#  El título de la figura es "Matriz de dispersión de  #
#  las variables, y la imagen se guarda con el nombre  #
#                 de "Dispersión.png".                 #
#                                                      #
########################################################

def dispersion(datos):
    n=datos.shape[1] #Tamaño de la base de datos
    k=0 #Inicialización de contador auxiliar que controla la posición de cada subfigura
    plt.figure(figsize=(7,7))
    for i in range(n): #Ciclo externo que controla el índice de una variable
        for j in range(n): #Ciclo interno que controla el índice de una variable
            k=k+1 #Incremento del contador auxiliar
            plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
            if(i==j):
                plt.subplot(n,n,k)
                plt.hist(datos[:,i],color='b',bins="auto") #Si "i==j", entonces se realiza un histograma en lugar de un gráfico de dispersión
            else:
                plt.subplot(n,n,k)
                plt.scatter(datos[:,i],datos[:,j],s=0.1,color='r') #Gráfico del histograma de la variable "i" con la variable "j"
    plt.suptitle("Matriz de dispersión de las variables")
    plt.savefig("Dispersión.png",dpi=300)
    plt.clf()

def grafica(datos,columnas,etiquetas=[],nombres=[],titulos=[],clasificacion=[],visualizacion="vertical",limites=[]):
    x=np.linspace(min(datos[:,0]),max(datos[:,0]),256)
    y=np.linspace(min(datos[:,1]),max(datos[:,1]),256)
    variables2d=[]
    for columna in columnas:
        z=griddata((datos[:,0],datos[:,1]),datos[:,columna],(x[None,:],y[:,None]),method='linear')
        variables2d.append(z)
    if(len(etiquetas)>0):
        colores=[]
        for etiqueta in etiquetas:
            z=griddata((datos[:,0],datos[:,1]),datos[:,etiqueta],(x[None,:],y[:,None]),method='nearest')
            colores.append(z)
    x,y=np.meshgrid(x,y)
    variables=[]
    for elemento in variables2d:
        variable=np.zeros(z.shape[0]*z.shape[1])
        k=0
        for j in range(z.shape[0]):
            for i in range(z.shape[1]):
                variable[k]=elemento[i,j]
                k=k+1
        variables.append(variable)
    if(len(etiquetas)>0):
        celdas=[]
        for color in colores:
            variable=np.zeros((z.shape[0]-1)*(z.shape[1]-1))
            k=0
            for j in range(z.shape[0]-1):
                for i in range(z.shape[1]-1):
                    if(np.isnan(color[i,j])):
                        variable[k]=255
                    else:
                        variable[k]=color[i,j]
                    k=k+1
            celdas.append(variable)
        colores=np.zeros((k,len(etiquetas)))
        for i in range(len(etiquetas)):
            colores[:,i]=celdas[i]
        colores.astype(int)
    malla=pv.StructuredGrid(x,y,variables2d[1])
    if(len(etiquetas)>0):
        malla_=pv.StructuredGrid(x,y,variables2d[1])
    malla.point_data[nombres[0]]=variables[0]
    if(len(etiquetas)>0):
        malla_.cell_data[nombres[1]]=colores
        contornos=malla.contour(10,scalars=nombres[0])
    if(len(clasificacion)>0):
        if(visualizacion=="vertical"):
            p=pv.Plotter(shape=(1,2))
        elif(visualizacion=="horizontal"):
            p=pv.Plotter(shape=(2,1))
    else:
        p=pv.Plotter()
    if(len(clasificacion)>0):
        actor=p.add_mesh(malla_,rgb=True)
        actor=p.add_mesh(contornos.copy(),opacity=0.7,cmap=plt.cm.jet,line_width=5)
    else:
        if(len(limites)>0):
            actor=p.add_mesh(malla,clim=limites,lighting=True,ambient=0.25,specular=0.25,cmap=plt.cm.jet)
        else:
            actor=p.add_mesh(malla,lighting=True,ambient=0.25,specular=0.25,cmap=plt.cm.jet)
    p.camera_position='xy'
    actor=p.show_bounds(xlabel="x [m]",ylabel="y [m]")
    p.camera.zoom(0.7)
    p.add_title(titulos[0],font_size=10,shadow=True)
    if(len(clasificacion)>0):
        if(visualizacion=="vertical"):
            p.subplot(0,1)
        elif(visualizacion=="horizontal"):
            p.subplot(1,0)
        mapa_=clasificacion[0]
        pivote=clasificacion[1]
        arreglo=np.zeros((pivote.shape[0],3))
        arreglo[:,0]=pivote[:,0]
        arreglo[:,1]=pivote[:,1]
        arreglo[:,2]=pivote[:,2]
        arreglo.astype(int)
        mapa_.cell_data["clases"]=arreglo
        actor=p.add_mesh(mapa_,lighting=True,ambient=0.25,specular=0.25,rgb=True)
        p.camera_position='xy'
        p.add_title(titulos[1],font_size=10,shadow=True)
        p.camera.zoom(1.5)
    p.show()

########################################################
#                                                      #
#     Gráfico del análisis en multiresolución en 1D    #
#    extraído de los ejemplos de la biblioteca pywt.   #
#    "data" es un arreglo de una dimensión al que se   #
# aplicará la transformada wavelet discreta, "w" es la #
# ondícula a usar, y "title" es el título de la figura #
#    resultante. Esta figura se muestra en pantalla    #
#                durante la ejecución.                 #
#                                                      #
########################################################

def plot_signal_decomp(data, w, title):
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    for i in range(5):
        (a, d) = pywt.dwt(a, w, pywt.Modes.periodization)
        ca.append(a)
        cd.append(d)
    rec_a = []
    rec_d = []
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))
    fig = plt.figure()
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)
    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))
    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))
    fig.set_figwidth(12)
    fig.set_figheight(10)
    plt.show()
    plt.clf()

#######################################################################
#                                                                     #
# Esta clase aplica filtros a dos conjuntos de datos                  #
# con formatos específicos: datos eléctricos y datos                  #
# magnetométricos.                                                    #
#                                                                     #
# Uso sugerido:                                                       #
#                                                                     #
# Eléctrica: inicializar primero y después aplicar uno de los dos     #
# métodos implementados.                                              #
# filtrado(archivo,"eléctrica"), donde "archivo" es la ruta al        #
# archivo donde se encuentran almacenados los datos, necesitan tener  #
# un formato  específico del "syscal" en donde las primeras dos       #
# columnas son el nivel y la posición estimada del punto de           #
# atribución.                                                         #
# filtrado.ondicula_electrica(voltaje,ondicula,componentes,recursion) #
# donde "voltaje" es la columna donde se encuentra la variable a      #
# filtrar. "ondicula" es la cadena que define a la ondícula a usar    #
# de acuerdo con las definiciones de pywt. "componentes" es el        #
# número de componentes para descomponer la señal original.           #
# "recursión" es el número de recursiones para filtrar cada nivel.    #
# Cada vez que se lleve a cabo una recursión se mostrará la gráfica   #
# de multiresolución, a partir de la cuál se deberán introducir de    #
# forma manual los factores de atenuación para cada componente de     #
# detalle.                                                            #
# filtrado.atipicos_electrica(logaritmo,metodo,vecinos,umbral)        #
# donde "logaritmo" es una variable de tipo booleano que especifica   #
# si los datos se analizarán en escala logarítmica. "metodo" puede    #
# "LOF" ("local outlier factor") o "IF" ("isolation forests").        #
# "vecinos" es el número de vecinos a usar para el método "LOF".      #
# "umbral", si se elige el método "LOF" será el cuantil para definir  #
# a un dato como anómalo, si se elige el método "IF" todo puntaje     #
# menor a esta variable será considerado como anomalía.               #
# Para aplicar los métodos implementados se usa una inmersión en 1D   #
# donde los nuevos puntos estarán definidos por el valor de           #
# resistividad (forzosamente guardado en la columna 7 del archivo)    #
# del punto i y por el valor de la resistividad del punto i+1.        #
# Cuando dos puntos consecutivos son etiquetados como anomalía,       #
# el dato en común de resistividad es eliminado de la base de datos   #
#                                                                     #
# Magnetometría: Inicializar y aplicar el método.                     #
# fitrado(archivo,"magnetometría"), donde "archivo" es la ruta al     #
# archivo donde se encuentran almacenados los datos, las dos primeras #
# columnas deben ser las coordenadas.                                 #
# filtrado.ondicula_magnetometria(columnas,resolucion_x,resolucion_y,malla,ventana,ondicula,componentes,pesos,pesos2) #
# "colummnas" es una tupla con los números de columna a filtrar, la   #
# numeración comienza desde cero. "resolucion_x" y "resolucion_y" es  #
# el número de elementos a interpolar para crear la malla. "malla" es #
# una variable booleana para especificar si se filtrará la malla      #
# completa. "ventana" es una variable booleana para especificar si se #
# filtrará una ventana definida por dos coordenadas                   #
# (superior izquierda e inferior derecha, en ese orden) introducidas  #
# de forma interactiva por el usuario a partir de una figura.         #
# "ondicula" es la cadena que define a la ondicula a usar de acuerdo  #
# con las definiciones de pywt. "componentes" es el número de         #
# componentes para descomponer la señal 2D original. "pesos" son los  #
# factores de atenuación de las componentes dadas por la transformada #
# wavelet, comienza por la componente de aproximación y               #
# posteriormente, en orden, las componentes de detalle. "pesos2" son  #
# los factores de atenuación de la malla cuando "malla=True" y        #
# "ventana=True"                                                      #
#                                                                     #
# filtrado.guardar(nombre), almacena los datos filtrados, para        #
# eléctrica almacena los datos con el mismo formato del archivo de    #
# entrada, para magnetometría se almacenan las coordenadas y las      #
# columnas filtradas. "nombre" es el nombre del archivo en el que     #
# se almacenarán los datos.                                           #
#                                                                     #
#######################################################################

class filtrado:
    def __init__(self,archivo,tipo):
        self.tipo=tipo
        d=open(archivo)
        self.datos=np.genfromtxt(d,delimiter=",",skip_header=1)
        d.close()
        if(tipo=="eléctrica"):
            self.niveles=int(max(self.datos[:,0])) #Se define el número de niveles
        elif(tipo=="magnetometría"):
            self.equis=np.zeros(0) #Se inicializa la variable de las coordenadas x de la ventana que elige el usuario en una figura
            self.ye=np.zeros(0) #Se inicializa la variable de las coordenadas y de la ventana que elige el usuario en una figura
        self.nd=self.datos.shape[0] #Tamaño de la base de datos
        self.nnd=self.datos.shape[1] #Número de columnas de la base de datos
    def evento_mouse(self,event): #Se obtienen las coordenadas de un punto a partir del click en una figura
        self.equis=np.append(self.equis,event.xdata)
        self.ye=np.append(self.ye,event.ydata)
    def ondicula_electrica(self,voltaje,ondicula="db2",componentes=5,recursion=1):
        modo=pywt.Modes.periodization #Se usa periodización para la transformada wavelet
        self.filtrados=np.zeros((self.nd,self.nnd))
        self.filtrados=self.datos #Se inicializan los datos filtrados igual a los datos originales
        for i in range(self.niveles): #Desplazamiento de los niveles
            nivel=i+1
            arreglo=self.datos[np.where(nivel==self.datos[:,0])] #Se separan los datos del nivel actual
            filtrar=arreglo[:,voltaje] #Se separa la columna a filtrar
            n=filtrar.shape[0] #Tamaño del nivel
            for iteracion in range(recursion): #Recursión
                if(i==0):
                    plot_signal_decomp(filtrar,ondicula,"Nivel "+str(i+1)) #Gráfica del análisis en multiresolución
                    atenuacion=np.zeros(componentes) #A partir de aquí se convierten los elementos de la tupla a un arreglo de numpy
                    for j in range(componentes):
                        cadena=input("Ingrese los factores de atenuación de la componente "+str(j+1)+": ")
                        atenuacion[j]=float(cadena)
                    atenuacion=np.flipud(atenuacion)
                    print()
                potencia=2 #A partir de aquí se encuentra el menor número potencia de 2 que es mayor al tamaño del nivel
                while potencia<n:
                    potencia=potencia*2
                pivote=np.zeros(potencia)
                pivote[0:n]=filtrar #Arreglo a filtrar cuyo tamaño es potencia de 2
                detalles=[]
                for componente in range(componentes): #Aplicación de la transformada wavelet
                    aproximacion,d=pywt.dwt(pivote,ondicula,modo)
                    detalles.append(d)
                    pivote=aproximacion
                detalles.reverse()
                for componente in range(componentes): #Aplicación de los pesos
                    if(atenuacion[componente]!=0):
                        aproximacion=pywt.idwt(aproximacion,detalles[componente]*atenuacion[componente],ondicula,modo)
                    else:
                        print(i)
                        aproximacion=pywt.idwt(aproximacion,None,ondicula,modo)
                filtrar=aproximacion[0:n]
            pivote=np.zeros(n)
            pivote=filtrar
            self.filtrados[np.where(nivel==self.datos[:,0]),voltaje]=pivote #Se igualan los datos originales a los datos filtrados
    def atipicos_electrica(self,logaritmo=True,metodo="LOF",vecinos=3,umbral=0.09):
        self.datos=self.datos[np.where(self.datos[:,10]>0)] #Se eliminan los datos negativos
        auxiliar=self.datos #Arreglo auxiliar que se inicializa con los datos originales
        auxiliar_anomalias=np.zeros(0) #
        for i in range(self.niveles): #Desplazamiento a través de los niveles
            nivel=i+1
            arreglo=self.datos[np.where(nivel==self.datos[:,0])] #Se extraen los datos del nivel actual
            filtrar=arreglo[:,6] #Se extraen los datos de la columna a filtrar
            if(logaritmo):
                filtrar=np.array(list(map(math.log10,filtrar))) #Se calcula el logaritmo de cada dato
            n=filtrar.shape[0] #Se obtiene el tamaño del nivel actual
            inmersion=np.zeros((n-1,2)) #Se define el arreglo de inmersión
            k=0 #Contador auxiliar para agregar valores al arreglo inmersión
            for j in range(n-1):
                inmersion[k,0]=filtrar[j]
                inmersion[k,1]=filtrar[j+1]
                k=k+1
            if(metodo=="LOF"):
                modelo=LOF(n_neighbors=vecinos)
                modelo.fit_predict(inmersion)
                factores=modelo.negative_outlier_factor_
                cuantil=np.quantile(factores,umbral) #Cálculo del cuantil
                indices=np.where(factores<=cuantil)
                anomalias=np.zeros(inmersion.shape[0]) #Es 0 para datos típicos
                anomalias[indices]=1 #Es 1 para datos atípicos
            elif(metodo=="IF"):
                modelo=IsFo(max_features=2)
                modelo.fit(inmersion)
                anomalias=modelo.predict(inmersion)
                anomalias[:]=0
                puntaje=modelo.decision_function(inmersion)
                for k in range(puntaje.shape[0]):
                    if(puntaje[k]<-umbral): #Se etiquetan datos anómalos con la puntuación y el umbral
                        anomalias[k]=1
            for k in range(anomalias.shape[0]): #Si hay dos datos atípicos seguidos se elimina el punto en común
                if(k>0 and k<anomalias.shape[0]-1):
                    if(anomalias[k]==1 and anomalias[k-1]==1 and anomalias[k+1]==1):
                        anomalias[k]=1
                    else:
                        anomalias[k]=0
            if(anomalias[-1]==1 and anomalias[-2]==0):
                anomalias=np.append(anomalias,np.array([1]))
            else:
                anomalias=np.append(anomalias,np.array([0]))
            auxiliar_anomalias=np.append(auxiliar_anomalias,anomalias) #Se agrega una columna con las etiquetas de datos atípicos
        auxiliar_anomalias=np.reshape(auxiliar_anomalias,(auxiliar_anomalias.shape[0],1))
        auxiliar=np.append(auxiliar,auxiliar_anomalias,axis=1)
        self.filtrados=auxiliar[np.where(auxiliar[:,-1]==0)] #Se eliminan los datos atípicos
        self.filtrados=np.delete(self.filtrados,-1,1) #Se elimina la columna de etiqueta
    def filtro_ventana_magne(self,columnas,resolucion_x,resolucion_y,ondicula,componentes,pesos):
        x=np.linspace(min(self.datos[:,0]),max(self.datos[:,0]),resolucion_x) #Datos en x
        y=np.linspace(min(self.datos[:,1]),max(self.datos[:,1]),resolucion_y) #Datos en y
        zz=np.zeros((resolucion_y,resolucion_x,0)) #Se crea el arreglo con los datos z de la mall apara cada columna
        for columna in columnas:
            z=griddata((self.datos[:,0],self.datos[:,1]),self.datos[:,columna],(x[None,:],y[:,None]),method='linear')
            z=np.reshape(z,(resolucion_y,resolucion_x,1))
            zz=np.append(zz,z,axis=2)
        fig=plt.figure()
        fig.canvas.mpl_connect('button_press_event',self.evento_mouse) #evento del click en la pantalla
        plt.contourf(x,y,zz[:,:,-1],16,cmap=plt.cm.jet) #Solo funciona para la última columna de entrada, esperando que sea gradiente por las unidades (abajo)
        plt.colorbar().set_label("nT/m")
        plt.title("Base de datos magnetometría columna "+str(columnas[-1]))
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.show()
        plt.clf()
        self.equis=np.array(list(self.equis)) #Se crea el arreglo de coordenadas en x para la ventana
        self.ye=np.array(list(self.ye))  #Se crea el arreglo de coordenadas en y para la ventana
        indices=np.zeros(4) #Arreglo con las 4 coordenadas necesarias, a partir del tercer click se ignora
        for i in range(x.shape[0]): #A partir de aquí se definen los índices para las ventanas
            if(x[i]>=self.equis[0]):
                indices[0]=i
                break
        for i in range(x.shape[0]):
            if(x[i]>=self.equis[1]):
                indices[1]=i
                break
        for i in range(y.shape[0]):
            if(y[i]>=self.ye[0]):
                indices[2]=i
                break
        for i in range(y.shape[0]):
            if(y[i]>=self.ye[1]):
                indices[3]=i
                break
        ventana=zz[int(indices[3]):int(indices[2]),int(indices[0]):int(indices[1]),-1] #Se aisla la ventana
        ventana=np.reshape(ventana,(ventana.shape[0],ventana.shape[1],1))
        for i in range(len(columnas)-1):
            piv=zz[int(indices[3]):int(indices[2]),int(indices[0]):int(indices[1]),i]
            piv=np.reshape(piv,(piv.shape[0],piv.shape[1],1))
            ventana=np.append(ventana,piv,axis=2)
        nv=ventana.shape[0] #Tamaño de la ventana
        nvn=ventana.shape[1]
        np2=2
        npn=2
        while np2<nv: #Obtención de la ventana de tamaño potencia de 2 para la transformada wavelet
            np2=np2*2
        while npn<nvn:
            npn=npn*2
        ventanap=np.zeros((np2,npn,len(columnas))) #Nueva ventana
        for i in range(len(columnas)):
            ventanap[0:nv,0:nvn,i]=ventana[:,:,i]
        for j in range(len(columnas)): #Transformada wavelet para cada columna
            coeficientes=pywt.mran(ventanap[:,:,j],wavelet=pywt.Wavelet(ondicula),level=componentes,transform="swtn")
            for i in range(len(coeficientes)): #Atenuación con los pesos
                if(i==0):
                    coeficientes[i]=coeficientes[i]*pesos[i]
                else:
                    coeficientes[i]["ad"]=coeficientes[i]["ad"]*pesos[i]
                    coeficientes[i]["dd"]=coeficientes[i]["dd"]*pesos[i]
                    coeficientes[i]["da"]=coeficientes[i]["da"]*pesos[i]
            ventanap[:,:,j]=pywt.imran(coeficientes) #Transformada wavelet inversa
        for i in range(len(columnas)): #Regreso de la ventana a su posición original dentro de la malla
            ventana[:,:,i]=ventanap[0:nv,0:nvn,i]
            zz[int(indices[3]):int(indices[2]),int(indices[0]):int(indices[1]),i]=ventana[:,:,i]
        plt.contourf(x,y,zz[:,:,-1],16,cmap=plt.cm.jet)
        plt.colorbar().set_label("nT/m")
        plt.title("Base de datos magnetometría columna "+str(columnas[-1]))
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.show()
        plt.clf()
        self.filtrados=np.zeros((0,len(columnas)+2)) #Se guardan los datos filtrados donde las primeras dos columnas son x y y
        for i in range(y.shape[0]):
            for j in range(x.shape[0]):
                vector=np.array([x[j],y[i]])
                for k in range(len(columnas)):
                    vector=np.append(vector,zz[i,j,k])
                vector=np.reshape(vector,(1,vector.shape[0]))
                self.filtrados=np.append(self.filtrados,vector,axis=0)
    def filtro_malla_magne(self,columnas,resolucion_x,resolucion_y,ondicula,componentes,pesos):
        x=np.linspace(min(self.datos[:,0]),max(self.datos[:,0]),resolucion_x)
        y=np.linspace(min(self.datos[:,1]),max(self.datos[:,1]),resolucion_y)
        zz=np.zeros((resolucion_y,resolucion_x,0))
        for columna in columnas:
            z=griddata((self.datos[:,0],self.datos[:,1]),self.datos[:,columna],(x[None,:],y[:,None]),method='linear')
            z=np.reshape(z,(resolucion_y,resolucion_x,1))
            zz=np.append(zz,z,axis=2)
        plt.contourf(x,y,zz[:,:,-1],16,cmap=plt.cm.jet)
        plt.colorbar().set_label("nT/m")
        plt.title("Base de datos magnetometría columna "+str(columnas[-1]))
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.show()
        plt.clf()
        for j in range(len(columnas)): #Se almacenan los índices con nan para igualarlos a cero
            indices_nan=np.zeros((0,2))
            for k in range(zz.shape[0]):
                for kk in range(zz.shape[1]):
                    if(np.isnan(zz[k,kk,j])):
                        vector=np.array([k,kk])
                        vector=np.reshape(vector,(1,2))
                        indices_nan=np.append(indices_nan,vector,axis=0)
                        zz[k,kk,j]=0
            coeficientes=pywt.mran(zz[:,:,j],wavelet=pywt.Wavelet(ondicula),level=componentes,transform="swtn")
            for i in range(len(coeficientes)):
                if(i==0):
                    coeficientes[i]=coeficientes[i]*pesos[i]
                else:
                    coeficientes[i]["ad"]=coeficientes[i]["ad"]*pesos[i]
                    coeficientes[i]["dd"]=coeficientes[i]["dd"]*pesos[i]
                    coeficientes[i]["da"]=coeficientes[i]["da"]*pesos[i]
            zz[:,:,j]=pywt.imran(coeficientes)
            for i in range(indices_nan.shape[0]):
                zz[int(indices_nan[i,0]),int(indices_nan[i,1]),j]=np.nan #Se regresa nan a las coordenadas que lo tenían originalmente
        plt.contourf(x,y,zz[:,:,-1],16,cmap=plt.cm.jet)
        plt.colorbar().set_label("nT/m")
        plt.title("Base de datos magnetometría columna "+str(columnas[-1]))
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.show()
        plt.clf()
        self.filtrados=np.zeros((0,len(columnas)+2))
        for i in range(y.shape[0]):
            for j in range(x.shape[0]):
                vector=np.array([x[j],y[i]])
                for k in range(len(columnas)):
                    vector=np.append(vector,zz[i,j,k])
                vector=np.reshape(vector,(1,vector.shape[0]))
                self.filtrados=np.append(self.filtrados,vector,axis=0)
    def ondicula_magnetometria(self,columnas,resolucion_x=128,resolucion_y=128,malla=True,ventana=False,ondicula="db4",componentes=3,pesos=[1,0.3,0.6,1],pesos2=[1,0.3,0.6,1]):
        if(ventana and not malla):
            self.filtro_ventana_magne(columnas,resolucion_x,resolucion_y,ondicula,componentes,pesos)
        elif(malla and not ventana):
            self.filtro_malla_magne(columnas,resolucion_x,resolucion_y,ondicula,componentes,pesos)
        elif(malla and ventana):
            self.filtro_ventana_magne(columnas,resolucion_x,resolucion_y,ondicula,componentes,pesos)
            self.datos=self.filtrados
            n=len(columnas)
            c=[]
            for i in range(n):
                c.append(i+2)
            self.filtro_malla_magne(c,resolucion_x,resolucion_y,ondicula,componentes,pesos2)
    def guardar(self,nombre):
        formato=[]
#        if(self.tipo=="magnetometría"):
#            filtrados=self.filtrados[:,0:2]
#            for i in range(self.filtrados.shape[1]):
#                formato.append("%0.8f")
#                if(i>1):
#                    x=self.datos[:,0]
#                    y=self.datos[:,1]
#                    f=griddata((self.filtrados[:,0],self.filtrados[:,1]),self.filtrados[:,i],(x[None,:],y[:,None]),method="nearest")
#                    f=np.reshape(f,(x.shape[0],1))
#                    filtrados=np.hstack((filtrados,f))
#            self.filtrados=filtrados
#        elif(self.tipo=="eléctrica"):
        for i in range(self.filtrados.shape[1]):
            formato.append("%0.8f")
        np.savetxt(nombre,self.filtrados,delimiter=",",fmt=formato)

########################################################
#                                                      #
# Esta clase prepara las bases de datos para el        #
# entrenamiento y clasificación, funciona para         #
# formatos de magnetometría y eléctrica.               #
#                                                      #
# preparar(carpeta), donde "carpeta" es la carpeta en  #
# la que se encuentran los archivos a preparar.        #
#                                                      #
# preparar.crear_base_magnetometria(vectores), donde   #
# "vectores" son las variables almacenadas en los      #
# archivos especificados en forma de lista.            #
#                                                      #
# preparar.crear_base_electrica(carpetas), donde       #
# "carpetas" es una lista con las carpetas de las      #
# inversiones resultantes usando el resipy.            #
#                                                      #
# preparar.vecindad(vecinos), donde "vecinos" es el    #
# número de elementos adyacenytes que se usarán en el  #
# entrenamiento.                                       #
#                                                      #
# preparar.guardar(nombre), donde "nombre" es el       #
# nombre del archivo                                   #
# donde se almacenará la base de datos que se usará    #
# para el entrenamiento.                               #
#                                                      #
########################################################

class preparar:
    def __init__(self,carpeta):
        self.carp=carpeta #Inicialización de la carpeta de trabajo
    def crear_base_magnetometria(self,vectores):
        carpeta=self.carp
        for i in range(len(vectores)): #Ciclo que desplaza cada elemento de la lista "vectores"
            if(i==0): #Primer ciclo, se almacenan las coordenadas únicamente
                d=open(carpeta+vectores[i],"r") #Se abre el archivo localizado en la carpeta de trabajo
                datos=np.genfromtxt(d,delimiter=",",skip_header=1)
                n=datos.shape[0] #Tamaño de la base de datos
                piv=np.zeros((n,1))
                self.variables=np.zeros((n,0))
                piv[:,0]=datos[:,0]
                self.variables=np.append(self.variables,piv,axis=1)
                piv[:,0]=datos[:,1]
                self.variables=np.append(self.variables,piv,axis=1)
                d.close()
            else:
                d=open(carpeta+vectores[i],"r")
                datos=np.genfromtxt(d,delimiter=",",skip_header=1)
                piv[:,0]=datos[:,2]
                self.variables=np.append(self.variables,piv,axis=1) #Se agregan como columna cada variable ingresada por nombre de archivo
                d.close()
        nans=np.zeros(n) #Se inicializa un arreglo auxiliar para el manejo de valores no numéricos
        np.isnan(self.variables[:,2],nans) #Se eliminan los vectores que no poseen un valor numérico
        nans=1-nans
        nans=np.array(nans,dtype=bool)
        self.variables=self.variables[nans,:]
    def crear_base_electrica(self,carpetas):
        carpeta=self.carp
        cadena="/wd/f001_res.dat" #Se usa la dirección de las inversiones realizadas en resipy
        for i in range(len(carpetas)):
            d=open(carpeta+carpetas[i]+cadena,"r")
            datos=np.genfromtxt(d,delimiter="   ")
            if(i==0): #Se inicializa con el primer archivo
                n=datos.shape[0]
                piv=np.zeros((n,1))
                self.variables=np.zeros((n,0))
                piv[:,0]=datos[:,0]
                self.variables=np.append(self.variables,piv,axis=1)
                piv[:,0]=datos[:,1]
                self.variables=np.append(self.variables,piv,axis=1)
                piv[:,0]=datos[:,2]
                self.variables=np.append(self.variables,piv,axis=1)
            else:
                z=griddata((datos[:,0],datos[:,1]),datos[:,2],(self.variables[:,0],self.variables[:,1]),method="linear") #Se interpolan los datos usando los puntos del primer vector
                piv[:,0]=z
                self.variables=np.append(self.variables,piv,axis=1)        
            d.close()
    def vecindad(self,vecinos):
        n=self.variables.shape[0] #Tamaño de la base de datos
        m=self.variables.shape[1]-2 #Número de variables explicativas
        pivote=self.variables[:,0:2] #Arreglo auxiliar con las coordenadas x,y o x,z
        piv=np.zeros((n,1)) #Arreglo auxiliar de tamaño igual al tamaño de la base de datos
        distancias=np.zeros((n,vecinos+1)) #Arreglo en el que se almacenarán las distancias entre cada vector con coordenadas x,y o x,z
        for i in range(n):
            indices=cercanos(pivote[i,:],pivote) #Se calculan los vectores más cercanos en cuanto a posición espacial
            distancias[i,:]=indices[0:vecinos+1] #Los resultados de la linea anterior se almacenan en un arreglo (índices)
        for i in range(m):
            j=i+2 #Se itera sobre las variables explicativas, se excluyen coordenadas
            piv[:,0]=self.variables[:,j] #Se inicializa una arreglo auxiliar igual a la variable explicativa j de cada vector
            pivote=np.append(pivote,piv,axis=1) #Se agrega la variable explicativa al arreglo auxiliar que almacenaba las coordenadas
            for k in range(vecinos):
                for ii in range(n):
                    piv[ii,0]=self.variables[int(distancias[ii,k+1]),j] #Se identifican los k vecinos más próximos de cada punto
                pivote=np.append(pivote,piv,axis=1) #Se agrega la variable explicativa correspondiente al arreglo auxiliar
        self.variables=pivote #Se iguala el arreglo auxiliar, con la información de los vecinos, al arreglo de salida
    def transformar(self,archivo=None,metodo="ecualizar",bins='auto',porcentaje=0.9):
        if("preparar.variables" in locals()):
            self.variables=self.variables
        else:
            d=open(self.carp+archivo)
            self.variables=np.genfromtxt(d,delimiter=",")
            d.close()
        if(metodo=="ecualizar"):
            for i in range(self.variables.shape[1]):
                if(i>1):
                    ecualizada=self.variables[:,i]
                    histograma,cajones=np.histogram(ecualizada,bins=bins)
                    maximo=max(ecualizada)
                    for j in range(ecualizada.shape[0]):
                        for k in range(cajones.shape[0]-1):
                            if(ecualizada[j]>=cajones[k] and ecualizada[j]<=cajones[k+1]):
                                break
                        suma=0
                        for l in range(k):
                            suma=suma+(histograma[l]/ecualizada.shape[0])
                        ecualizada[j]=suma*maximo
                    self.variables[:,i]=ecualizada
        elif(metodo=="logaritmica"):
            for i in range(self.variables.shape[1]):
                if(i>1):
                    transformada=self.variables[:,i]
                    for j in range(transformada.shape[0]):
                        transformada[j]=np.log10(transformada[j])
                    self.variables[:,i]=transformada
    def guardar(self,nombre):
        m=self.variables.shape[1] #Número de variables de la base de datos
        formato=[]
        for i in range(m):
            formato.append("%0.8f")
        np.savetxt(nombre,self.variables,delimiter=",",fmt=formato) #Almacenamiento

######################################################################
#                                                                    #
# Esta clase realiza una serie de modelos inversos                   #
# geoeléctricos semi-automáticos cuyo fin es ser una                 #
# base de datos para entrenamiento.                                  #
#                                                                    #
# modelar_electrica_2D(archivo,mallai,mallaf), donde                 #
# "archivo" es la ruta al archivo con formato vtk que                #
# se usará como base para la creación de los modelos                 #
# (nota para mi, recuerdo que no es el grid que se                   #
# almacena cuando le pones "guardar", debe ser el                    #
# generado por el software en otra carpeta, debes                    #
# recordar cuál es, busca en donde sea concordante el                #
# número de regiones y su resistividad).                             #
# "mallai" es el archivo que define todo el comienzo                 #
# del archivo vtk, hasta antes de SCALARS res0 y                     #
# LOOKUP_TABLE. "mallaf" es el archivo que define todo               #
# el final del archivo vtk, hasta SCALARS phase0.                    #
#                                                                    #
# modelar_electrica_2D.geometria(nelectrodos,dx),                    #
# donde "nelectrodos" es el número de electrodos                     #
# espaciados cada "dx".                                              #
#                                                                    #
# modelar_electrica_2D.modelo(secuencia,nelectrodos,dx), donde       #
# "secuencia" es la ruta al archivo que contiene la                  #
# secuencia de lectura con encabezados C+,C-,P+,P-, "nelectrodos" es #
# el número de electrodos de la secuencia y "dx" es su separación.   #
#                                                                    #
# modelar_electrica_2D.modelos(intervalos,secuencias,n_modelos),     #
# donde "intervalos" es una lista de dos dimensiones con los         #
# límites inferior y superior para la generación de                  #
# resistividades aleatorias, "secuencias" es una lista con las       #
# direcciones de las secuencias que se usarán para realizar los      #
# modelos, "n_modelos" es el número de modelos que se                #
# realizarán. Se recibirán muchas respuestas del API de resipy.      #
#                                                                    #
# modelar_electrica_2D.agrupar_secuencias(vecinos,iteracion,isecuencia), #
# donde "vecinos" es el número de vectores que se desean agregar     #
# para formar parte de las variables explicativas, "iteracion"       #
# es el número de iteraciones modelos realizados, por omisión se     #
# toman los usados en "modelos" (si aplica), isecuencia es el        #
# número de secuencias usadas, por omisión se toman las usadas       #
# en "modelos", (si aplica).                                         #
#                                                                    #
# modelar_electrica_2D.agrupar_modelo(iteracion,isecuencia,vecinos), #
# donde "iteracion", "isecuencia" y "vecinos" tienen el mismo        #
# comportamiento que las variables usadas en "agrupar_secuencias",   #
# la base de datos resultante se almacena en un archivo nombrado     #
# "Modelos.dat".                                                     #
#                                                                    #
# ####################################################################

class modelar_electrica_2D:
    def __init__(self,archivo,mallai,mallaf):
        lector=vtk.vtkDataSetReader() #Inicialización del lector de archivos vtk
        lector.SetFileName(archivo) #Se carga el archivo vtk de entrada
        lector.ReadAllScalarsOn() #Lectura de los datos escalares (resistividad y región son las que no sinteresan)
        lector.Update()
        datos=lector.GetOutput() #Se guardan los datos en la variable
        celdas=datos.GetCellData() #En celdas se almacena la información de datos
        reg=celdas.GetArray('region') #En reg se guardan los datos de region
        self.nd=datos.GetNumberOfCells() #Almacenamiento del número de celdas
        self.regiones=np.array([reg.GetValue(i) for i in range(self.nd)]) #Conversión de los valores de región a arreglo de numpy, con el fin de cotejar la región de cada celda con su valor de resistividad
        self.nreg=np.unique(self.regiones) #Obtención del número de regiones
        print("\n",len(self.nreg),"regiones\n")
        with open(mallai,"r") as archivo:
            self.inicio=archivo.read() #Se carga la primer parte del archivo vtk
        with open(mallaf,"r") as archivo:
            self.fin=archivo.read() #Se carga la segunda parte del archivo vtk, se espera únicamente agregar valores de resistividad para generar un nuevo archivo
    def geometria(self,nelectrodos,dx):
        self.electrodos=np.zeros((nelectrodos,3)) #Inicialización de la geometría de los electrodos
        self.electrodos[:,0]=np.arange(0,nelectrodos*dx,dx) #Posición de los electrodos
    def modelo(self,secuencia,nelectrodos,dx):
        proyecto=Project(dirname="tmp",typ="R2") #Inicialización del proyecto en resipy
        self.geometria(nelectrodos,dx)
        proyecto.setElec(self.electrodos) #Definición de electrodos
        proyecto.importMesh("modelar.vtk") #Importación de Malla a usar para la inversión
        proyecto.importSequence(secuencia) #Se importa la secuencia a usar para la inversión
        proyecto.forward(noise=0.0) #Modelación directa sin ruido
        proyecto.invert() #Inversión del modelo directo creado
        d=open("tmp/invdir/f001_res.dat") #Se abre el archivo generado
        self.rho=np.genfromtxt(d) #Almacenamiento de la resistividad
        d.close()
    def modelos(self,intervalos,secuencias,n_modelos,nelectrodos,dx):
        resistividades=np.zeros(self.nd) #Inicialización de arreglo para almacenar resistividades
        self.nmodelos=n_modelos
        self.nsecuencias=len(secuencias)
        for ijk in range(n_modelos): #Ciclo que itera sobre el número de modelos
            for region in self.nreg: #Ciclo que itera sobre el número de regiones
                for i in range(self.nd): #Ciclo que itera sobre el número de celdas
                    if(self.regiones[i]==region): #Si la región de la celda es igual a la región iterada
                        resistividades[i]=random.uniform(intervalos[int(region)-1][0],intervalos[int(region)-1][1]) #Se calcula una resistividad aleatoria dentro de un intervalo
            cadena=np.array2string(resistividades,precision=8,separator="     ",threshold=self.nd+1,max_line_width=self.nd+1) #El arreglo de resistividades se convierte a cadena
            with open("modelar.vtk","w") as archivo:
                archivo.write(self.inicio+cadena[1:-1]+self.fin) #Se almacena un archivo vtk con las resistividades calculadas y el inicio y fin de un archivo vtk
            isecuencia=-1 #Inicialización de número auxiliar de secuencias
            for secuencia in secuencias: #Ciclo que itera sobre la cantidad de secuencias
                isecuencia=isecuencia+1
                self.modelo(secuencia,nelectrodos,dx) #Se realiza la modelación usando la secuencia en cuestión
                self.rho=np.delete(self.rho,3,1) #Se elimina la columna con logaritmo
                np.savetxt("Secuencia-"+str(isecuencia)+"_modelo-"+str(ijk),self.rho,delimiter=', ',header="X, Y, "+str(isecuencia),fmt=['%0.8f','%0.8f','%0.8f']) #Almacenamiento de los resultados del modelo y secuencia actual
            subprocess.run(["rm","modelar.vtk"]) #Eliminación del archivo vtk usado para modelar
    def agrupar_secuencias(self,vecinos,iteracion=0,isecuencia=0):
        if(iteracion==0 and isecuencia==0): #Carga de los números de modelos y secuencias usados, con el fin de usar datos de una ejecución independiente
            iteracion=self.nmodelos
            isecuencia=self.nsecuencias
        self.vec=vecinos
        for i in range(iteracion): #Ciclo que itera sobre el número de modelos
            formato=[]
            for j in range(isecuencia): #Ciclo que itera sobre el número de secuencias
                d=open("Secuencia-"+str(j)+"_modelo-"+str(i)) #Apertura del archivo de acuerdo con el número de modelo y número de secuencia
                datos=np.genfromtxt(d,delimiter=",",skip_header=1) #Los datos dek archivo se almacenan
                d.close()
                n=datos.shape[0] #Tamaño de la base de datos actual
                if(j==0): #Si es la primer secuencia se inicializan los arreglos
                    datos_secuencia=np.zeros((n,0)) #Inicialización de variable para almacenar los datos de las secuencias para un modelo fijo
                    datos_secuencia=np.append(datos_secuencia,datos[:,0:2],axis=1) #Se agregan los datos espaciales
                    formato.append('%0.8f')
                    formato.append('%0.8f')
                pivote=np.zeros((n,1)) #Inicialización de arreglo auxiliar...
                pivote[:,0]=datos[:,2] #... que se iguala a los datos de resistividad almacenados anteriormente
                datos_secuencia=np.append(datos_secuencia,pivote,axis=1) #Se agregan los datos de resistividad al arreglo de salida
                formato.append('%0.8f')
                if(vecinos>0): #Si se desea agregar información sobre los puntos vecinos:
                    datos_secuencia=np.append(datos_secuencia,np.zeros((n,vecinos)),axis=1) #Se agrega a la variable de salida un arreglo de tamaño igual al número de puntos vecinos deseados
                    for k in range(vecinos):
                        formato.append('%0.8f')
                    piv=datos_secuencia[:,0:2] #Arreglo auxiliar que almacena las coordenadas de cada vector
                    distancias=np.zeros((n,n)) #Inicialización de matriz de "distancias"
                    for k in range(n): #Ciclo que itera sobre el número de vectores, se obtienen los indices de los vectores más cercanos en orden ascendente
                        indices=cercanos(piv[k,:],piv)
                        distancias[k,:]=indices
                    for k in range(vecinos): #Ciclo que itera sobre el número de vecinos
                        for l in range(n): #Ciclo que itera sobre el número de vectores
                            datos_secuencia[l,datos_secuencia.shape[1]-int(vecinos)+k]=datos[int(distancias[l,k+1]),2] #Se iguala cada vecino de cada vector al valor de resistividad correspondiente
                subprocess.run(["rm","Secuencia-"+str(j)+"_modelo-"+str(i)]) #Se elimina el archivo leido
            np.savetxt("Modelo-"+str(i),datos_secuencia,delimiter=', ',fmt=formato) #Almacenamiento de un nuevo archivo con las secuencias agrupadas a forma de columna
    def agrupar_modelo(self,iteracion=0,isecuencia=0,vecinos=0):
        if(iteracion==0 and isecuencia==0 and vecinos==0): #Carga de los números de modelos, secuencias y vecinos usados, con el fin de usar datos de una ejecución independiente
            iteracion=self.nmodelos
            isecuencia=self.nsecuencias
            vecinos=self.vec
        formato=[]
        j=(int(vecinos)+1)*int(isecuencia)+2 #Cálculo del número de columnas que se obtendrá, depende del número de vecinos y de las secuencias realizadas (coordenadas también)
        entrenamiento=np.zeros((0,j)) #Inicialización de la base de datos final
        for i in range(j): 
            formato.append('%0.8f')
        for i in range(iteracion): #Ciclo que itera sobre el número de modelos
            d=open("Modelo-"+str(i))
            datos=np.genfromtxt(d,delimiter=",") #Se lee cada archivo de modelo realizado
            entrenamiento=np.append(entrenamiento,datos,axis=0) #Se agrega cada elemento de todos los modelos como un nuevo vector
            d.close()
        np.savetxt("Modelos.dat",entrenamiento,delimiter=', ',fmt=formato) #Almacenamiento de la base de datos

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

#########################################################
#                                                       #
# Esta clase aplica la clasificación a partir del       #
# entrenamiento anterior, así como la preparación de    #
# las mallas para visualización de los centroides o de  #
# la red neuronal. entrenamiento.clasificar almacena    #
# la base de datos en todo el proceso.                  #
#                                                       #
# clasificacion(archivo,columnas), donde "archivo" es   #
# la ruta al archivo con la base de datos a             #
# clasificar, "columnas" es una lista con los números   #
# de columna que indican las variables a usar para      #
# clasificar, se espera sean las mismas usadas en el    #
# entrenamiento. clasificacion.x son las coordenadas    #
# en x, clasificacion.y son las coordenadas en y,       #
# clasificacion.nn es el tamaño de la base de datos.    #
#                                                       #
# clasificacion.reduccion_pca(columnas,estandarizaciones,pcas,componentes), #
# donde "columnas" es una lista que contiene los        #
# conjuntos de variables a reducir, esperando que sean  #
# los mismos usados en el entrenamiento.                #
# "estandarizaciones" son las transformaciones usadas   #
# para estandarizar los datos de entrenamiento. "pcas"  #
# contiene la matriz de transformación de cada          #
# conjunto de variables durante el entrenamiento.       #
# "componentes" es una lista con el número de           #
# componentes usadas para reducir la dimensionalidad,   #
# se espera que coincida con las usadas para el         #
# entrenamiento.                                        #
#                                                       #
# clasificacion.escalamiento(escala), donde "escala"    #
# es la transormación usada en los datos de             #
# entrenamiento.                                        #
#                                                       #
# clasificacion.kmeans(kmeans), donde "kmeans" es el    #
# objeto resultado del entrenamiento.                   #
# clasificacion.etiquetas son las etiquetas de cada     #
# dato, clasificacion.centros son las coordenadas de    #
# los centroides. A la salida genera un histograma con  #
# la frecuencia de uso de los centroides.               #
#                                                       #
# clasificacion.kmedians(kmedians), donde "kmedians"    #
# es el objeto resultado del entrenamiento.             #
# clasificacion.etiquetas son las etiquetas de cada     #
# dato, clasificacion.centros son las coordenadas de    #
# los centroides. A la salida genera un histograma con  #
# la frecuencia de uso de los centroides.               #
#                                                       #
# clasificacion.kmedoids(kmedoids), donde "kmedoids"    #
# es el objeto resultado del entrenamiento.             #
# clasificacion.etiquetas son las etiquetas de cada     #
# dato, clasificacion.centros son las coordenadas de    #
# los centroides. A la salida genera un histograma con  #
# la frecuencia de uso de los centroides. Aún no es     #
# aplicable por cuestiones de memoria                   #
#                                                       #
# clasificacion.som(som), donde "som" es el objeto      #
# resultante del levantamiento. "clasificacion.indices  #
# es un arreglo con el identificador de cada neurona,   #
# clasificacion.etiquetas son las etiquetas de cada     #
# dato, clasificacion.pesos son los pesos de cada       #
# neurona de la red, clasificacion.nx es el número de   #
# filas de la red, clasificacion.ny es el número de     #
# columnas en la red, clasificacion.etiquetas son las   #
# etiquetas de cada dato. A la salida genera un         #
# histograma con la frecuencia de uso de las neuronas.  #
#                                                       #
# clasificacion.etiquetar(columna), agrega a            #
# clasificacion.clasificados las coordenadas, valor     #
# original de la variable "columna", valor              #
# pre-procesado de la variable "columna" y las          #
# etiquetas -para cada dato- esperando que sean las     #
# etiquetas de color RGB.                               #
#                                                       #
# clasificacion.malla(metodo,radio,cumulos,metodo_sub). #
# Donde "metodo" es el usado para el entrenamiento y    #
# clasificación, pudiendo ser "centroides" (para        #
# métodos aglomerativos), "red_rectangular" o           #
# "red_hexagonal". "radio" es el radio de las esferas   #
# para el despliegue de los centroides para el método   #
# "centroides". "cumulos" es el número de conjuntos de  #
# neuronas deseado para el cálculo de colores en la red #
# neuronal. "metodo_sub" es el método a usar para la    #
# generación de los conjuntos de neuronas, pudiendo ser #
# "kmedias", "kmedianas" o "kmedoides".                 #
# clasificacion.malla_etiquetas contiene la malla ([0]) #
# y los valores de color para cada celda ([1]).         #
#                                                       #
#########################################################

class clasificacion:
    def __init__(self,archivo,columnas,escalamiento_min=[],escalamiento_max=[]):
        d=open(archivo)
        self.originales=np.genfromtxt(d,delimiter=",") #Se almacenan los datos originales
        d.close()
        d=open(archivo)
        self.clasificar=np.genfromtxt(d,delimiter=",") #Lectura de los datos
        d.close()
        self.x=self.clasificar[:,0]
        self.x=np.reshape(self.x,(self.x.shape[0],1)) #Se almacena la coordenada en x
        self.y=self.clasificar[:,1]
        self.y=np.reshape(self.y,(self.y.shape[0],1)) #Se almacena la coordenada en y
        n=self.clasificar.shape[1] #Número de variables explicativas
        mascara=np.zeros(n,dtype=bool)
        mascara[columnas]=True
        self.clasificar=self.clasificar[:,mascara] #Se separan las variables explicativas (columnas) que se van a usar
        self.nn=self.clasificar.shape[0] #Tamaño de la base de datos
        n=self.clasificar.shape[1] #Número de variables explicativas
        if(len(escalamiento_min)>0 and len(escalamiento_max)>0 and len(escalamiento_min)==len(escalamiento_max)):
            for i in range(n):
                pivote=self.clasificar[:,i]
                rmin=min(pivote)
                rmax=max(pivote)
                for j in range(pivote.shape[0]):
                    pivote[j]=(pivote[j]-rmin)/(rmax-rmin)*(escalamiento_max[i]-escalamiento_min[i])+escalamiento_min[i]
                self.clasificar[:,i]=pivote
    def reduccion_pca(self,columnas,estandarizaciones,pcas,componentes):
        sin_cambios=np.arange(self.clasificar.shape[1]) #Variables explicativas originales
        npo=len(pcas) #Número de componentes principales
        for columna in columnas:
            for digito in columna:
                if digito in sin_cambios:
                    sin_cambios=np.delete(sin_cambios,np.where(sin_cambios==digito)[0]) #Variables a las que no se aplicará pca
        conjunto=self.clasificar[:,sin_cambios] #Conjunto con las variables que no se reducirán
        for i in range(npo):
            arreglo=estandarizaciones[i].transform(self.clasificar[:,columnas[i]]) #Estandarización de las variables
            arreglo=np.matmul(arreglo,np.transpose(pcas[i].components_[0:int(componentes[i]),:])) #Aplicación de pca
            arr=arreglo.reshape((self.nn,int(componentes[i])))
            conjunto=np.append(conjunto,arr,axis=1) #Se agregan los datos reducidos dimensionalmente
        self.clasificar=conjunto #Se obtiene el conjunto reducido dimensionalmente
    def escalamiento(self,escala):
        self.clasificar=escala.transform(self.clasificar) #Escalamiento de la base de datos
    def malla(self,metodo="centroides",radio=1.,cumulos=2,metodo_sub="kmedoides"):
        if(metodo=="centroides"):
            nc=self.centros.shape[0] #Número de centroides
            iso=manifold.Isomap(n_neighbors=2,n_components=3) #Reducción a dimensionalidad 3 usando isomap
            iso.fit(self.centros)
            self.centros=iso.transform(self.centros)
#            print(self.centros)
            colores=np.zeros((nc,3)) #Arreglo para almacenar los colores
            etiquetas2d=np.zeros((0,3)) #Arreglo que define el color para cada celda
            pivote=np.zeros((1680,3)) #Arreglo auxiliar que almacena las celdas de una esfera
            dh=1/(nc+0.01) #Delta de matiz
            h=-dh #Inicialización de el matiz
            radios=np.zeros(nc)
            for i in range (nc):
                for j in range(self.nn):
                    if(int(self.etiquetas[j,0])==i):
                        radios[i]=radios[i]+1
            minimo=min(radios)
            maximo=max(radios)
            for i in range(nc):
                radios[i]=((radios[i]-minimo)/(maximo-minimo)*9.)+1.
                if(radios[i]==1):
                    radios[i]=1.0001
                radios[i]=math.log10(radios[i])*radio
#            print(radios)
            for i in range(nc):
                if(i==0):
                    plano=pv.Sphere(center=(self.centros[i,0],self.centros[i,1],self.centros[i,2]),radius=radios[i]) #Inicialización de la malla con el primer centroide
                else:
                    esfera=pv.Sphere(center=(self.centros[i,0],self.centros[i,1],self.centros[i,2]),radius=radios[i]) #Esfera del siguiente centroide
                    plano=plano+esfera #Se agrega la esfera a la malla
                h=h+dh #Incremento en el matiz
                rgb=np.array(colorsys.hsv_to_rgb(h,0.75,0.9)) #Obtenición del color rgb con la matiz, siendo la intensidad y satuzación fijos
                colores[i,0]=round(rgb[0]*255) #Canal rojo
                colores[i,1]=round(rgb[1]*255) #Canal verde
                colores[i,2]=round(rgb[2]*255) #Canal azul
                pivote[:,:]=colores[i,:] #Se iguala el color a la esfera pivote
                etiquetas2d=np.vstack((etiquetas2d,pivote)) #Se agregan las celdas de la esfera actual
            pivote=np.zeros((self.nn,3)) #Arreglo pivote para definir las nuevas etiquetas
            for i in range(self.nn):
                pivote[i,:]=colores[int(self.etiquetas[i,0]),:] #Obtenición de las etiquetas con color
            self.etiquetas=pivote #Nuevas etiquetas
        elif(metodo=="red_hexagonal" or metodo=="red_rectangular"):
            if(metodo_sub=="kmedias"): #Se realiza la aglomeración de las neuronas para definir subconjuntos
                etiquetas=[] #Inicialización de arreglo para almacenar las neuronas que se parecen entre sí
                kmedias=KMeans(n_clusters=cumulos).fit(self.pesos) #Aplicación de k-medias
                etiquetasp=kmedias.labels_ #Se almacenan las etiquetas obtenidas
                for i in range(cumulos):
                    cumulo=[] #Se inicializa un arreglo para almacenar las neuronas que pertenecen al mismo centroide
                    for j in range(self.pesos.shape[0]):
                        if(etiquetasp[j]==i):
                            cumulo.append(j) #Se agrega la neurona al arreglo del centroide actual
                    etiquetas.append(cumulo) #Se agrega el conjunto de neuronas del centroide actual al arreglo de etiquetas
            elif(metodo_sub=="kmedianas"):
                columnas=self.pesos.shape[1]
                centros=[]
                for i in range(cumulos):
                    arreglo=np.random.rand(columnas)
                    centros.append(arreglo)
                kmedianas=kmedians(self.pesos,centros)
                kmedianas.process()
                etiquetas=kmedianas.get_clusters()
            elif(metodo_sub=="kmedoides"):
                etiquetas=[]
                kmedoides=KMedoids(n_clusters=cumulos,method="pam",init="k-medoids++").fit(self.pesos)
                etiquetasp=kmedoides.labels_
                for i in range(cumulos):
                    cumulo=[]
                    for j in range(self.pesos.shape[0]):
                        if(etiquetasp[j]==i):
                            cumulo.append(j)
                    etiquetas.append(cumulo)
            etiquetas2d=np.zeros((self.nx*self.ny,3)) #Se inicializa arreglo para almacenar los colores de cada celda
            dh=1/(cumulos+0.01) #Delta de matiz
            h=dh/2 #Inicialización del matiz
            for cumulo in etiquetas: #Ciclo que itera sobre cada centroide
                dc=0.9/len(cumulo) #Delta saturación
                h=h+dh #Actualización de matiz
                s=1 #Inicialización de saturación
                for elemento in cumulo: #Ciclo que itera sobre las neuronas que pertenecen al centroide actual
                    s=s-dc #Actualización de saturación
                    v=1 #valor de intensidad
                    rgb=np.array(colorsys.hsv_to_rgb(h,s,v)) #Conversión de HSI a RGB
                    etiquetas2d[int(elemento),0]=round(rgb[0]*255) #Canal rojo
                    etiquetas2d[int(elemento),1]=round(rgb[1]*255) #Canal verde
                    etiquetas2d[int(elemento),2]=round(rgb[2]*255) #Canal azul
            pivote=np.zeros((self.nn,3)) #Arreglo auxiliar para definir las nuevas etiquetas
            for i in range(self.nn):
                pivote[i,:]=etiquetas2d[int(self.etiquetas[i,0]),:] #Obtenición de las etiquetas con color
            self.etiquetas=pivote #Nuevas etiquetas
            if(metodo=="red_hexagonal"):
                hexagonop=np.array([[0,0,0.1],[0,-1,0.1],[0.866,-1.5,0.1],[1.732,-1,0.1],[1.732,0,0.1],[0.866,0.5,0.1],[0,0,0],[0,-1,0],[0.866,-1.5,0],[1.732,-1,0],[1.732,0,0],[0.866,0.5,0]]) #Hexágono base
                hexagono=np.zeros((hexagonop.shape[0],hexagonop.shape[1])) #Arreglo para almacenar nuevas coordenadas de hexágonos desplazados
                hexagono[:,:]=hexagonop[:,:] #Inicialización de arreglo de hexágono móvil
                puntos=np.empty((0,3)) #Inicialización del arreglo que almacenará los vértices de todos los hexágonos
                for i in range(self.nx): #Ciclo que itera sobre la dirección vertical de la red neuronal
                    if(i>0): #Cuando no se trata de la primera fila
                        if(i%2==1): #Si se trata de una fila impar
                            hexagonop[:,0]=hexagonop[:,0]+0.866 #Se desplazan las coordenadas horizontales hacia la derecha
                            hexagonop[:,1]=hexagonop[:,1]-1.5 #Se desplazan las coordenadas verticales
                            hexagono[:,:]=hexagonop[:,:] #El nuevo hexágono que se desplazará
                        else: #Si se trata de una fila par
                            hexagonop[:,0]=hexagonop[:,0]-0.866 #Se desplazan las coordenadas horizontales hacia la izquierda
                            hexagonop[:,1]=hexagonop[:,1]-1.5 #Se desplazan las coordenadas verticales
                            hexagono[:,:]=hexagonop[:,:] #El nuevo hexágono que se desplazará
                    for j in range(self.ny): #Ciclo que itera sobre la dirección horizontal de la red neuronal
                        if(i==0 and j==0): #Si es la primer neurona
                            arreglo=np.arange(12) #Número de puntos que formarán al hexágono
                            celdas=np.hstack(([12],arreglo)) #Se agrega la celda
                            tipo=np.array([vtk.VTK_HEXAGONAL_PRISM]) #Se agrega el tipo de celda
                        else: #Si no es la primer neurona
                            arreglo=arreglo+12 #Se obtiene los puntos de la siguiente neurona
                            celdas=np.hstack((celdas,[12])) #Se agrega la celda
                            celdas=np.hstack((celdas,arreglo))
                            tipo=np.hstack((tipo,[vtk.VTK_HEXAGONAL_PRISM])) #Se agrega el tipo de celda
                        puntos=np.vstack((puntos,hexagono)) #Se agregan los puntos de la celda
                        hexagono[:,0]=hexagono[:,0]+1.732 #Se desplaza horizontalmente la celda
                plano=pv.UnstructuredGrid(celdas,tipo,puntos) #Se define la malla no estructurada
            elif(metodo=="red_rectangular"):
                x=np.linspace(0,self.nx,self.nx+1) #Número de puntos en las celdas horizontales
                y=np.linspace(0,self.ny,self.ny+1) #Número de púntos en las celdas verticales
                x,y=np.meshgrid(y,x) #Mallado de puntos para coordenadas
                z=np.zeros((self.nx+1,self.ny+1)) #Se desea obtener una malla plana
                malla=pv.StructuredGrid(x,y,z) #Malla rectangular
                plano=malla.flip_y(inplace=False) #Se voltea la malla para cambiar la definición de celdas
        self.malla_etiquetas=[] #Arreglo para almacenar la malla y los valores de cada celda
        self.malla_etiquetas.append(plano)
        self.malla_etiquetas.append(etiquetas2d)
    def kmeans(self,kmeans):
        self.etiquetas=kmeans.predict(self.clasificar) #Obtenición de etiquetas
        self.etiquetas=np.reshape(self.etiquetas,(self.etiquetas.shape[0],1)) #Cambio a dimensión (n,1)
        self.centros=kmeans.cluster_centers_ #Coordenadas de los centroides
        nc=self.centros.shape[0]
        histograma=np.zeros((nc,2))
        formato=["%d","%0.8f"]
        for i in range(nc):
            histograma[i,0]=i+1
            for j in range(self.nn):
                if(int(self.etiquetas[j,0])==i):
                    histograma[i,1]=histograma[i,1]+1
        for i in range(nc):
            if(histograma[i,1]==0):
                histograma[i,1]=0
            else:
#                histograma[i,1]=math.log10(histograma[i,1])
                histograma[i,1]=histograma[i,1]
        np.savetxt("Histograma.dat",histograma,delimiter=" ",fmt=formato)
        n,bins,patches=plt.hist(x=self.etiquetas[:,0],bins="auto",color="#0504aa",alpha=0.7,rwidth=0.9)
        plt.grid(axis="y",alpha=0.5)
        plt.xlabel("Centroide")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de los centroides / k-means")
        plt.savefig("Histograma.png")
        plt.clf()
    def kmedians(self,kmedians):
        self.etiquetas=np.zeros((self.nn,1)) #Inicialización de las etiquetas
        self.centros=np.array(kmedians.get_medians()) #Coordenadas de los centroides
        for i in range(self.nn):
            indices=cercanos(self.clasificar[i,:],self.centros) #Se busca el centroide más cercano a cada dato
            self.etiquetas[i,0]=indices[0] #Se etiqueta al dato
        nc=self.centros.shape[0]
        histograma=np.zeros((nc,2))
        formato=["%d","%0.8f"]
        for i in range(nc):
            histograma[i,0]=i+1
            for j in range(self.nn):
                if(int(self.etiquetas[j,0])==i):
                    histograma[i,1]=histograma[i,1]+1
        for i in range(nc):
            if(histograma[i,1]==0):
                histograma[i,1]=0
            else:
                histograma[i,1]=math.log10(histograma[i,1])
        np.savetxt("Histograma.dat",histograma,delimiter=" ",fmt=formato)
        n,bins,patches=plt.hist(x=self.etiquetas[:,0],bins="auto",color="#0504aa",alpha=0.7,rwidth=0.9)
        plt.grid(axis="y",alpha=0.5)
        plt.xlabel("Centroide")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de los centroides / k-medians")
        plt.savefig("Histograma.png")
        plt.clf()
    def kmedoids(self,kmedoids): #Aún no aplicable por cuestiones de memoria
        self.etiquetas=kmedoids.predict(self.clasificar) #Obtención de las etiquetas
        self.etiquetas=np.reshape(self.etiquetas,(self.etiquetas.shape[0],1)) #Cambio a dimensión (n,1)
        self.centros=kmedoids.cluster_centers_ #Coordenadas de los centroides
        nc=self.centros.shape[0]
        histograma=np.zeros((nc,2))
        formato=["%d","%0.8f"]
        for i in range(nc):
            histograma[i,0]=i+1
            for j in range(self.nn):
                if(int(self.etiquetas[j,0])==i):
                    histograma[i,1]=histograma[i,1]+1
        for i in range(nc):
            if(histograma[i,1]==0):
                histograma[i,1]=0
            else:
                histograma[i,1]=math.log10(histograma[i,1])
        np.savetxt("Histograma.dat",histograma,delimiter=" ",fmt=formato)
        n,bins,patches=plt.hist(x=self.etiquetas[:,0],bins="auto",color="#0504aa",alpha=0.7,rwidth=0.9)
        plt.grid(axis="y",alpha=0.5)
        plt.xlabel("Centroide")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de los centroides / k-medoids")
        plt.savefig("Histograma.png")
        plt.clf()
    def som(self,som,topologia="rectangular"):
        mapa=som.distance_map() #Obtención del mapa de distancias
        self.indices=np.arange(mapa.shape[0]*mapa.shape[1]) #Se obtiene el número de neuronas
        self.indices=np.reshape(self.indices,(mapa.shape[0],mapa.shape[1])) #Cambio a dimensión (n,m)
        self.etiquetas=np.zeros((self.nn,1)) #Inicialización de etiquetas
        self.pesos=som.get_weights() #Obtención de los pesos de las neuronas
        self.nx=self.pesos.shape[0] #Número de filas en la red
        self.ny=self.pesos.shape[1] #Número de columnas en la red
        self.pesos=np.reshape(self.pesos,(self.nx*self.ny,self.pesos.shape[2])) #Cambio a dimensiones para coincidir con la red
        for i in range(self.nn):
            neurona=som.winner(self.clasificar[i,:]) #Obtención de la neurona ganadora
            self.etiquetas[i,0]=self.indices[int(neurona[0]),int(neurona[1])] #Se etiqueta al dato
        nc=self.indices.shape[0]
        histograma=np.zeros((nc,2))
        formato=["%d","%0.8f"]
        for i in range(nc):
            histograma[i,0]=i+1
            for j in range(self.nn):
                if(int(self.etiquetas[j,0])==i):
                    histograma[i,1]=histograma[i,1]+1
        for i in range(nc):
            if(histograma[i,1]==0):
                histograma[i,1]=0
            else:
                histograma[i,1]=histograma[i,1]
        np.savetxt("Histograma.dat",histograma,delimiter=" ",fmt=formato)
        n,bins,patches=plt.hist(x=self.etiquetas[:,0],bins="auto",color="#0504aa",alpha=0.7,rwidth=0.9)
        plt.grid(axis="y",alpha=0.5)
        plt.xlabel("Centroide")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de las neuronas / SOM")
        plt.savefig("Histograma.png")
        plt.clf()
        if(topologia=="rectangular"):
            frecuencias=som.activation_response(self.clasificar)
            plt.pcolor(frecuencias,cmap="inferno_r")
            plt.colorbar()
            plt.title("Hit-map")
            plt.savefig("SOM_hits.png")
            plt.clf()
        elif(topologia=="hexagonal"):
            frecuencias=som.activation_response(self.clasificar)
            xx, yy=som.get_euclidean_coordinates() #A partir de aquí se hace la graficación de la matriz de distancias, algoritmo tomado de la documentación de la biblioteca minisom
            weights=som.get_weights()
            f=plt.figure(figsize=(10,10))
            ax=f.add_subplot(111)
            ax.set_aspect('equal')
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    wy=yy[(i,j)]*np.sqrt(3)/2
                    hexa=RegularPolygon((xx[(i,j)],wy),numVertices=6,radius=.95/np.sqrt(3),facecolor=plt.cm.inferno_r(frecuencias[i,j]),edgecolor='black')
                    ax.add_patch(hexa)
            xran=np.arange(weights.shape[0])
            yran=np.arange(weights.shape[1])
            plt.xticks(xran-.5,xran)
            plt.yticks(yran*np.sqrt(3)/2,yran)
            divider=make_axes_locatable(plt.gca())
            ax_cb=divider.new_horizontal(size="5%",pad=0.05)
            cb1=colorbar.ColorbarBase(ax_cb,cmap=cm.inferno_r,orientation='vertical')
            cb1.ax.get_yaxis().labelpad=int(7*7)
            cb1.ax.set_ylabel('Número de aciertos',rotation=270, fontsize=16)
            plt.gcf().add_axes(ax_cb)
            plt.savefig('SOM_hits.png')
            plt.clf()
    def etiquetar(self,columna):
        aux=self.originales[:,columna+2] #Arreglo auxiliar con la columna de la base original
        aux=np.reshape(aux,(aux.shape[0],1))
        self.clasificados=np.append(self.x,self.y,axis=1) #Se agrega al arreglo de salida las coordenadas x y y
        self.clasificados=np.append(self.clasificados,aux,axis=1) #Se agrega al arreglo de salida la columna de la base original
        escala=preprocessing.StandardScaler().fit(aux) #Escala
        aux=escala.transform(aux) #Aplicación
#        aux=self.clasificar[:,columna] #Arreglo auxiliar con la columna de la base pre-procesada
        aux=np.reshape(aux,(aux.shape[0],1))
        self.clasificados=np.append(self.clasificados,aux,axis=1) #Se agrega al arreglo de salida la columna de la base pre-procesada
        self.clasificados=np.append(self.clasificados,self.etiquetas,axis=1) #Se agrega al arreglo de salida las etiquetas, esperando sean las etiquetas de color RGB.
