import math
import pywt
import numpy as np
import pyvista as pv
from minisom import MiniSom
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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

########################################################
#                                                      #
# Cálculo de error topológico para un SOM de geometría #
#                      hexagonal                       #
#                                                      #
# "datos": Base de datos a agrupar, dimensiones nxm,   #
#          n - número de vectores                      #
#          m - número de características               #
#                                                      #
# "som": obejeto minisom                               #
#                                                      #
########################################################

def error_topologico_hexagonal(datos,som):
    n=datos.shape[0]
    coordenadas=som.get_euclidean_coordinates()
    error=0
    dist=math.sqrt(1.25) #Distancia entre centros de los hexágonos
    for i in range(n):
        ganadora=list(som.winner(datos[i,:]))
        mapa=som._activation_map
        for j in range(mapa.shape[0]): #Se inicializa una neurona que no sea la ganadora
            for k in range(mapa.shape[1]):
                cs=[j,k]
                if(ganadora!=cs):
                    break
            if(ganadora!=cs):
                break
        segunda=mapa[cs[0],cs[1]]
        for j in range(mapa.shape[0]): #Se encuentra la segunda neurona ganadora
            for k in range(mapa.shape[1]):
                c=[j,k]
                if(mapa[j,k]<segunda and c!=ganadora):
                    segunda=mapa[j,k]
                    cs=c
        distancia=metrica(np.array([coordenadas[0][ganadora[0]][ganadora[1]],coordenadas[1][ganadora[0]][ganadora[1]]]),np.array([coordenadas[0][cs[0]][cs[1]],coordenadas[1][cs[0]][cs[1]]]),2) #Se calcula la distancia de la neurona a la neurona ganadora
        if(distancia>dist): #Se evalua si las neuronas son vecinas
            error=error+1
    return error/n #Cálculo del error

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

###########################################################################################################################
#                                                                                                                         #
# Creación de los gráficos finales                                                                                        #
#                                                                                                                         #
# "datos": Base de datos a graficar, las primeras dos columnas son las coordenadas x,y;                                   #
#                                                                                                                         #
# "columnas": elementos para definir la malla, la primer columna es la variable a graficar, la segunda es la 'topografía' #
#             de la malla (coordenada z)                                                                                  #
#                                                                                                                         #
# "etquetas": columnas que traen la información de los colores RGB, un canal por columna en ese mismo órden, es opcional, #
#             se aplica cuando se desea visualizar la agrupación                                                          #
#                                                                                                                         #
# "nombres": lista que contiene los nombres de las variables de las columnas especificadas                                #
#                                                                                                                         #
# "titulos": lista que contiene los títulos que aparecerán en la gráfica, tanto la imagen agrupada como del mapa de la    #
#            agrupación                                                                                                   #
#                                                                                                                         #
# "clasificacion": objeto salida del método 'clasificacion.malla_etiquetas'                                               #
#                                                                                                                         #
# "visualizacion": 'vertical' si se desea visualizar magnetometría, 'horizontal' si se desea visualizar eléctrica         #
#                                                                                                                         #
# "limites": lista con los limites superior e inferior para acotar el dominio de graficación                              #
#                                                                                                                         #
###########################################################################################################################

def grafica(datos,columnas,etiquetas=[],nombres=[],titulos=[],clasificacion=[],visualizacion="vertical",limites=[]):
    x=np.linspace(min(datos[:,0]),max(datos[:,0]),256) #Dominio en x en 256 elementos
    y=np.linspace(min(datos[:,1]),max(datos[:,1]),256) #Dominio en y en 256 elementos
    variables2d=[] #Inicialización de la lista que contiene las variables en 2D para cada columna
    for columna in columnas: #Interpolación de las variables usando el dominio x,y
        z=griddata((datos[:,0],datos[:,1]),datos[:,columna],(x[None,:],y[:,None]),method='linear')
        variables2d.append(z)
    if(len(etiquetas)>0): #Se hace lo propio para los colores RGB, opcional
        colores=[]
        for etiqueta in etiquetas:
            z=griddata((datos[:,0],datos[:,1]),datos[:,etiqueta],(x[None,:],y[:,None]),method='nearest')
            colores.append(z)
    x,y=np.meshgrid(x,y)
    variables=[] #Inicialización de una lista que guarda las variables pero en 1D
    for elemento in variables2d:
        variable=np.zeros(z.shape[0]*z.shape[1])
        k=0
        for j in range(z.shape[0]):
            for i in range(z.shape[1]):
                variable[k]=elemento[i,j]
                k=k+1
        variables.append(variable)
    if(len(etiquetas)>0): #Se hace lo propio para los colores RGB, opcional
        celdas=[]
        for color in colores:
            variable=np.zeros((z.shape[0]-1)*(z.shape[1]-1))
            k=0
            for j in range(z.shape[0]-1):
                for i in range(z.shape[1]-1):
                    if(np.isnan(color[i,j])): #Se evalua si el color interpolado es nan, en ese caso se lleva al valor 255
                        variable[k]=255
                    else:
                        variable[k]=color[i,j]
                    k=k+1
            celdas.append(variable)
        colores=np.zeros((k,len(etiquetas)))
        for i in range(len(etiquetas)):
            colores[:,i]=celdas[i]
        colores.astype(int)
    malla=pv.StructuredGrid(x,y,variables2d[1]) #Se crea la malla base usando las variables de la segunda columna
    if(len(etiquetas)>0): #Si se va a agregar una subgráfica se crea otra malla concordante con la anterior, a fin de agregar las otras variables
        malla_=pv.StructuredGrid(x,y,variables2d[1])
    malla.point_data[nombres[0]]=variables[0] #Se agrega la variable de la primer columna
    if(len(etiquetas)>0): #Si se va a agregar la subgráfica, se añaden los colores como variables y se crean los contornos de la primer variable
        malla_.cell_data[nombres[1]]=colores
        contornos=malla.contour(10,scalars=nombres[0])
    if(len(clasificacion)>0): #De ser requerido se divide el área de graficación
        if(visualizacion=="vertical"):
            p=pv.Plotter(shape=(1,2))
        elif(visualizacion=="horizontal"):
            p=pv.Plotter(shape=(2,1))
    else:
        p=pv.Plotter() #Se inicializa la graficación
    if(len(clasificacion)>0): #De ser requerido se agregan los colores y los contornos de la primer variable
        actor=p.add_mesh(malla_,rgb=True)
        actor=p.add_mesh(contornos.copy(),opacity=0.7,cmap=plt.cm.jet,line_width=5)
    else: #Si no se requieren se añade únicamente el mapa de la primer variable con base a los limites establecidos
        if(len(limites)>0):
            actor=p.add_mesh(malla,clim=limites,lighting=True,ambient=0.25,specular=0.25,cmap=plt.cm.jet)
        else:
            actor=p.add_mesh(malla,lighting=True,ambient=0.25,specular=0.25,cmap=plt.cm.jet)
    p.camera_position='xy' #Se establece la posición del visualizador en el plano xy
    actor=p.show_bounds(xlabel="x [m]",ylabel="y [m]") #Se etiquetan los ejes coordenados
    p.camera.zoom(0.7) #Se establece el zoom
    p.add_title(titulos[0],font_size=10,shadow=True) #Se añade el primer título
    if(len(clasificacion)>0): #Se crea el subgráfico si se requiere
        if(visualizacion=="vertical"):
            p.subplot(0,1)
        elif(visualizacion=="horizontal"):
            p.subplot(1,0)
        mapa_=clasificacion[0] #Malla con la información del SOM o de los centroides
        pivote=clasificacion[1] #Colores de las celdas en RGB
        arreglo=np.zeros((pivote.shape[0],3)) #Creación del arreglo con los colores RGB
        arreglo[:,0]=pivote[:,0]
        arreglo[:,1]=pivote[:,1]
        arreglo[:,2]=pivote[:,2]
        arreglo.astype(int)
        mapa_.cell_data["clases"]=arreglo #Se añaden los colores como variables
        actor=p.add_mesh(mapa_,lighting=True,ambient=0.25,specular=0.25,rgb=True)
        p.camera_position='xy'
        p.add_title(titulos[1],font_size=10,shadow=True)
        p.camera.zoom(1.5)
    p.show() #Se muestra el gráfico

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
