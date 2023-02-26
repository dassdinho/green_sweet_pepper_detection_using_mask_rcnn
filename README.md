# Green Sweet Pepper Detection Using Mask R-CNN in Greenhouses Documentation

Esta es una implementación de Mask R-CNN en Python 3.6, Keras 2.1.2, y TensorFlow-GPU 1.4.0 desarrollada por el equipo de Matterport (https://github.com/matterport/Mask_RCNN)
para la detección del fruto y pedúnculo del pimiento dulce/bell pepper en sus variedades de color más comunes 
(verde, rojo, amarillo y naranja), enfocándose en pimiento dulce verde, esta siendo la variedad de color más complicada
debido a su parecido con el fondo de su entorno real de producción.

![b (4)](https://user-images.githubusercontent.com/107544707/205514983-d2b25fe5-1ae3-400f-bee9-ce1a12cda57e.png)

## Requisitos

- Se utiliza Windows 10 en esta implementación específica.
- Anaconda (esta implementación utiliza un entorno virtual de anaconda).
  - https://www.anaconda.com
- CUDA Toolkit 8.0.61/8.0 GA2 y cuDNN v6.0 para CUDA 8.0 (si cuentas con una GPU NVIDIA para aceleramiento de hardware).
  - CUDA Toolkit Archive: https://developer.nvidia.com/cuda-toolkit-archive
    - Producto versión directa: https://developer.nvidia.com/cuda-80-ga2-download-archive
    - Documentación: https://docs.nvidia.com/cuda/archive/8.0/
  - cuDNN Archive: https://developer.nvidia.com/rdp/cudnn-archive
    - [Buscar] Download cuDNN v6.0 (April 27, 2017), for CUDA 8.0
- Visual Studio 2015 o Visual Studio Community 2015.

## Visión general

- Paso 1: Clonar este repositorio.
- Paso 2: Creación del conda virtual environment en base al archivo YAML que contiene todas las dependencias necesarias.
- Paso 3: Instalar pycocotools.
- Paso 4: Instalar Mask R-CNN (modelo-código, scripts principales).
- Paso 5: Comprobar correcta instalación de dependencias.
- Paso 6: Probarlo.

## Proceso

Ejecutar estos comandos en una Anaconda Prompt.

### Paso 1 - Clonar este repositorio

Dentro de una carpeta específica para este entorno (recomendable), ejecutar este comando:

`https://github.com/dassdinho/green_sweet_pepper_detection_using_mask_rcnn.git`

### Paso 2 - Creación del conda virtual environment en base al archivo YAML con todas las dependencias

- Ingresar a la carpeta de **Mask R-CNN**

`cd <PATH del directorio>`

 Ejemplo: `cd F:\mask_rcnn_models\green_sweet_pepper_detection_using_mask_rcnn\Mask_RCNN`
 
- Crear el entorno en base al archivo YAML

`conda env create -f <PATH del archivo>`

 Ejemplo: `conda env create -f F:\mask_rcnn_models\green_sweet_pepper_detection_using_mask_rcnn\MASK_RCNN_env_v1.yaml`
 
 - Activar el entorno
 
 Cada vez que deseemos utilizar el entorno en una terminal/anaconda prompt es necesario activarlo (ya dentro de la carpeta Mask R-CNN).
 
 `conda activate MASK_RCNN_env_v1`
 
 De la misma manera, podemos desactivar el entorno para que no esté disponible desde dicha terminal.
 
 `conda deactivate`
 
 ### Paso 3 - Instalar pycocotools
 
 Una vez dentro de la carpeta Mask R-CNN y con el entorno activo...
 
 - **NOTA**: pycocotools requiere Visual C++ 2015 Build Tools.
  - Descargar si es necesario: https://go.microsoft.com/fwlink/?LinkId=691126
    - También puedes seguir las instrucciones del repositorio original: https://github.com/philferriere/cocoapi
 - Ya dentro del entorno activo, usamos pip para instalar pycocotools
 
 `pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`
 
 ### Paso 4 - Instalar el modelo de Mask R-CNN
 
  Una vez dentro de la carpeta Mask R-CNN y con el entorno activo...
 
 Ejecutar el siguiente comando para instalar el modelo de Mask R-CNN en nuestro entorno (***mask-rcnn***):
 
 `python setup.py clean --all install`
  
 De igual manera, este comando lo debemos de ejecutar siempre y cuando queramos ver reflejados nuestros cambios realizados en los scripts contenidos
 en la carpeta de ***mask-rcnn*** (scripts principales que definen al modelo y sus herramientas).
 
 ### Paso 5 - Comprobación de dependencias
 
 Antes de conocer qué puede hacer esta implementación de Mask R-CNN, debemos comprobar que todo lo que instalamos está en su lugar.
 Para eso, utilizamos el siguiente comando y buscamos los nombres de todas las dependencias que acabamos de instalar (***pycocotools*** y ***mask-rcnn***):
 
 `pip list`
 
 ![h23](https://user-images.githubusercontent.com/107544707/205516823-58813193-75d4-490b-8419-d4e416e75aa9.png)

 ### Paso 6 - Vamos a probarlo!
 
 En este punto, explicaremos brevemente los archivos relevantes para entrenar y visualizar nuestro modelo personalizado de Mask R-CNN, para un mejor entendimiento de los códigos recomendados el repositorio original (https://github.com/matterport/Mask_RCNN):
 
 ### Info en repositorio original
 
 ![cc](https://user-images.githubusercontent.com/107544707/205518250-03d400f3-792e-48a4-9425-ec1f7e9e51ba.JPG)

 #### Carpeta *annotation_tool*
 
 En el directorio principal (*Mask R-CNN*) existe la carpeta *annotation_tool*, la cual contiene la herramienta (en una versión específica) utilizada para realizar las anotaciones de nuestras máscaras de segmentación de instancias.
 La herramienta en cuestión es la VIA VGG Image Annotator (https://www.robots.ox.ac.uk/~vgg/software/via/).
 
 #### Carpeta *mrcnn*
 
 Dentro de la carpeta *mrcnn*, en el directorio principal, podremos modificar la configuración del modelo y herramientas de visualización de los resultados.
 
 ![bb](https://user-images.githubusercontent.com/107544707/205518263-e66c7ca3-bf59-46b1-a088-bbd5f5390784.JPG)
 
 #### Carpeta *dataset*
 
 Dentro de la carpeta *dataset*, en el directorio principal, podremos agregar nuestras imágenes y archivos JSON con las anotaciones que generemos para el entrenamiento del modelo.
 
 La información de la dataset utilizada la puedes encontrar en la siguiente liga: https://drive.google.com/drive/folders/1EcMkmG8q7ZoA0WM5FOjpXURkvLe2UoGr
 
 #### Carpeta *logs*
 
 Dentro de la carpeta *logs*, en el directorio principal, podremos encontrar los archivos .H5 que contienen los pesos de nuestros modelos en cada epoch (archivo que define a nuestro modelo entrenado listo para realizar inferencia/predicciones).
 
 Los archivos .H5 generados en esta implementación para pimiento dulce se encuentran en la siguiente liga: https://drive.google.com/drive/folders/1EcMkmG8q7ZoA0WM5FOjpXURkvLe2UoGr
  
 #### Carpeta *samples*
 
 Dentro de la carpeta *samples*, en el directorio principal, encontramos todas las Jupyter Notebooks utilizas para comprender el modelo, las bases de datos utilizadas y los reultados obtenidos a través de distintos ejemplos que se describen en el repositorio original.
 
 En este caso específico, nuestra implementacióna para frutos y pedúnculos del pimiento dulce se encuentra en la carpeta *pepper*, en la cual encontramos lo siguiente:
 - ***pepper_dass.py***: Básicamente, el script necesario para entrenar la implementación de Mask R-CNN con bases de datos propias, dónde podemos especificar hiperparámetros y configurar todos los aspectos de nuestro modelo para el objetivo requerido.
 
 - ***inspect_pepper_data_dass.ipynb***: Esta libreta nos permite visualizar paso a paso el proceso que sigue el modelo de Mask R-CNN para obtener las máscaras de segmentación de instancias en nuestras propias bases de datos.
 
 - ***inspect_pepper_model_dass.ipynb***: Esta libreta nos permite visualizar los resultados finales de las métricas, máscaras, objeto detectado, clase definida y bounding box de nuestro modelo personalizado de Mask R-CNN.
 
 ![cc (2)](https://user-images.githubusercontent.com/107544707/205518299-f0a9c71a-927b-49ee-bace-76551d30e6bc.JPG)
 
*De esta manera, cualquier error o duda emergente es muy probable que exista en el apartado de Issues del repositario original.*

### ***Únicamente queda comenzar con la libreta de Jupyter demo.ipynb y ejecutarla.***
 
## List@s? Vamos!
