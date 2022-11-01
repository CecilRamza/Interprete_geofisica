# Intérprete geofísica

## Universidad Nacional Autónoma de México
## Instituto de Investigaciones en Matemáticas aplicadas y en Sistemas

## Metodologías de interpretación automática de datos geofísicos multidimensionales
## Manuel Ortiz Osio
## Correo: soff.rpg@gmail.com
## Director de tesis: Dr. Erik Molino Minero Re

*Código implementado para el trabajo de* [Tesis de maestría](https://tesiunam.dgb.unam.mx/F/IYUEVQKCRXV4B74YRNE4AC22RJH5FYNJUNJ2XAHSB4U12DFDJA-07798?func=full-set-set&set_number=020124&set_entry=000001&format=999 "TESIUNAM")

Contiene lo siguiente:

- **`filrado.py`:** código para realizar el filtrado de datos de magnetometría 2D y de tomografía de resistividad eléctrica 2D (*TRE2D*).
- **`preproceso.py`:** prepara los datos para el entrenamiento.
- **`entrenamiento.py`:** código que realiza el entrenamiento no supervizado.
- **`agrupamiento.py`:** se encarga de aplicar las redes o centroides para realizar un agrupamiento de vectores de magnetometría 2D o de una *TRE2D*.
- ***`herramientas.py:`*** contiene funciones necesarias para el funcionamiento general del sistema implementado.

Además de ejemplos de uso para cada elemento.

***

## Filtrado

### Requerimientos

- `pywt`
- `math`
- `numpy`
- `matplotlib`
- `scipy`
- `sklearn`
- `herramientas.py`

### Uso

Se trata de una sola clase que aplica filtros basados en la transformada de ondícula y algoritmos de detección de anomalíasa datos de magnetometría 2D y de *TRE2D*. La entrada es un archivo ascii separado por comas y con un encabezado con estructura matricial `nxm,`donde `n` es el número de vectores y `m` es el número de características. Para magnetometría se espera que las primeras dos columnas sean las coordenadas espaciales de cada vector; para *TRE2D* se espera que la primer columna sea el nivel de profundidad y la segunda la coordenada en `x` del punto de atribución, así mismo se espera que los vectores se encuentren ordenados por nivel de forma ascendente. Consultar el archivo [Ejemplo_filtrado.py](https://github.com/CecilRamza/Interprete_geofisica/blob/main/Ejemplo_filtrado.py "Ejemplos del filtrado implementado").

***

## Preproceso

### Requerimientos

- `vtk`
- `random`
- `subprocess`
- `numpy`
- `resipy`
- `scipy`

### Uso

La clase `preparar` recibe como argumento la ruta a las bases de datos que se usarán para entrenar o agrupar. Para magnetometría se reciben los nombres de los archivos que se quieran preparar, se considera un archivo ascii separado por comas con un encabezado y 3 columnas; mientras que para *TRE2D* se reciben las carpetas en donde se encuentran los archivos de salida de la inversión de datos, considerando que este proceso se realizó con la biblioteca `resipy`. Esta clase se encarga de encontrar los vecinos espacialmente más cercanos y de aplciar transformaciones.

La clase `modelar_electrica_2D` recibe los elementos inicial y final de las mallas vtk creadas con la biblioteca `resipy`, así como la malla en su totalidad incluyendo los valores de resistividad. Posteriormente se modela usando valores de resistividad aleatorios dentro de un intervalo definido usando secuencias de lectura con la estructura requerida por la biblioteca `resipy`, así mismo se requiere de la geometría del levantamiento. Finalmente se agrupan los modelos realizados en un solo archivo.

Consultar el archivo [Ejemplo_preparacion.py](https://github.com/CecilRamza/Interprete_geofisica/blob/main/Ejemplo_preparacion.py "Ejemplos para preparar las bases de datos").

***




## Entrenamiento

### Requerimientos

- `random`
- `numpy`
- `minisom`
- `matplotlib`
- `sklearn`
- `pyclustering`
- `mpl_toolkits`
- `herramientas.py`

### Uso

Se trata de una sola clase, se reciben los archivos ascii separados por comas sin encabezados, y una lista con las columnas con las que se desea entrenar. Esta clase contiene los métodos para reducir la dimensionalidad mediante *PCA*, escalamiento y normalización de los datos, el entrenamiento mediante métodos de agrupamiento y el entrenamiento mediante *SOM*. Consultar el archivo [Ejemplo_entrenamiento.py](https://github.com/CecilRamza/Interprete_geofisica/blob/main/Ejemplo_entrenamiento.py "Ejemplos para aplicar entrenamiento").

***

## Agrupamiento

### Requerimientos

- `vtk`
- `math`
- `colorsys`
- `numpy`
- `pyvista`
- `minisom`
- `matplotlib`
- `sklearn`
- `pyclustering`

### Uso

Esta clase realiza el agrupamiento a la base de datos ingresada, aplicándole el respectivo escalamiento y reducciones usadas en el conjunto de entrenamiento. Consultar el archivo [Ejemplo_agrupamiento.py](https://github.com/CecilRamza/Interprete_geofisica/blob/main/Ejemplo_agrupamiento.py "Ejemplos de despliegue de resultados de agrupamiento") para ejemplos en el despliegue de resultados usando `pyvista`.
