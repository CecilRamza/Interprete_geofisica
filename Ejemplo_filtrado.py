from filtrado import *

###
# Filtrado de datos de TRE2D usando transformada de ondícula
###

##
# archivo.csv es un archivo es un archivo nxm separado por comas y un encabezado
##

columna=7
filtro=filtrado("archivo.csv","eléctrica")
filtro.ondicula_electrica(columna,ondicula="db2",componentes=5,recursion=1)

#
# A continuación se ingresan los pesos a aplicar para cada componente de detalle
#

filtro.guardar("archivo_filtrado.csv")

###
# Retiro de vectores atípicos de una TRE2D
# Se puede usar después del filtro anterior
###

##
# La primer columna de archivo.csv debe ser el nivel del punto de atribución. Se espera que los datos se encuentren agrupados por nivel ascendente
##

filtro=filtrado("archivo.csv","eléctrica")
filtro.atipicos_electrica(columna,logaritmo=True,metodo="LOF",vecinos=3,umbral=0.9)
filtro.guardar("archivo_filtrado.csv")

###
# Filtrado de datos de magnetometría 2D usando transformada de ondícula
###

columnas=[2,3,4]
filtro=filtrado("archivo.csv","magnetometría")
filtro.ondicula_magnetometria(columnas,resolucion_x=32,resolucion_y=32,malla=True,ventana=True,ondicula="db4",componentes=3,pesos=[0,0.9,0.4,0],pesos2=[1,0.3,0.6,1])

##
# A continuación se da click en la esquina superior izquierda y esquina inferior derecha de la ventana a filtrar
##

filtro.guardar("archivo_filtrado.csv")
