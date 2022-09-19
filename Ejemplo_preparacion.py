from preproceso import *

###
# Modelación semi-automática de TRE2D
###

##
# Malla.vtk es la malla donde se encuentra el modelo base, las mallas inicio y fin son las porciones inicial y final de dicho archivo excluyendo los datos de resistividad
# intervalos son los rangos de variación de cada región definida en el modelo base
# secuencias son los archivos que contiene la secuencia de adquisición para generar los modelos, con el formato requerido por resIPy
# modelos es el número de modelos a generar
# electrodos es el número de electrodos por cada secuencia
# dx es la separación entre electrodos
##

intervalos=[[0.01,10],[10,100],[100,1000]]
secuencias=["WS.csv","DD.csv"]
modelos=10
electrodos=24
dx=5.0
vecinos=3
modelacion=modelar_electrica_2D("Malla.vtk","Malla_inicio.vtk","Malla_fin.vtk")
modelacion.modelos(intervalos,secuencias,modelos,electrodos,dx)
modelacion.agrupar_secuencias(vecinos)
modelacion.agrupar_modelo()

###
# Preparación de bases de datos para el entrenamiento
###

##
# carpeta es la dirección en donde se localiza la base de datos a preparar
# datos son los archivos que contienen las bases de datos que se desean agrupar y preparar, cada archivo contiene la información de una variable explicativa
##

carpeta_magnetometria="Magnetometría/"
carpeta_electrica="Eléctrica/"
datos_magnetometria=["BOT.csv","TOP.csv","GRAD.csv","BOT_RP.csv","TOP_RP.csv","BOT_SA.csv","TOP_SA.csv","BOT_GH.csv","TOP_GH.csv","BOT_TD.csv","TOP_TD.csv","BOT_GHTD.csv","TOP_GHTD.csv"]
datos_electrica=["WS","DD"]
vecinos_magnetometria=4
vecinos_electrica=3
preparados_magnetometria=preparar(carpeta_magnetometria)
preparados_magnetometria.crear_base_magnetometria(datos_magnetometria)
preparados_magnetometria.vecindad(vecinos_magnetometria)
preparados_magnetometria.transformar(metodo="ecualizar",bins="auto")
preparados_magnetometria.guardar("Malla_mag.csv")
preparados_electrica=preparar(carpeta_electrica)
preparados_electrica.crear_base_electrica(datos_electrica)
preparados_electrica.vecindad(vecinos_electrica)
preparados_electrica.transformar(metodo="logaritmica")
preparados_electrica.guardar("TRE.csv")
