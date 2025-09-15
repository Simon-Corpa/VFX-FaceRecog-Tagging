# Script para VFX de face recognition & tagging
Este script se creo con la intención de incluirlo en el vídeo documental [Sumergidos en Atlántico](https://www.youtube.com/watch?v=JyahuAla-Dk) de Periferia Centro.
La versión definitiva del vídeo no incluye el efecto para no sobrecargar la cadencia del montaje.

Mediante [Face_recog-detectfunctions.py](/Face_recog-detectfunctions.py) puedes detectar las caras de un vídeo de input para después limpiar la base de datos generada y utilizarla como input en el segundo script, [Face_recog-printfunctions.py](/Face_recog-printfunctions.py) para generar overlays que sobreponer sobre el vídeo original con información de las caras seleccionadas.

![Frame de prueba](/Test_frame.jpg)

# Planteamiento del proceso
El script buscaba tratar clips donde apareciese el público del concierto con la idea de profundizar en el hecho de que componía las influencias del mismo trabajo que presentaban en directo. Para ello, generé el código, procesé clips y preparé una carpeta con todas las caras que aparecían y una tabla rellenable con los IDs de esas caras para que rellenasen información automatizable de aquellas caras que fuesen familiares y/o relevantes. 

Ello permitía incluir a los artistas dentro del proceso de elaboración del vídeo y facilitaba la elaboración del efecto encorsetando la forma de rellenar la información para facilitar la posterior lectura y procesamiento de la tabla.
## Face_recog-detectfunctions.py 👥
Este script de Python es una herramienta para la detección, almacenamiento y gestión de rostros a partir de archivos de video. Permite organizar los resultados en una estructura de carpetas de proyecto, almacenar los datos de los rostros en un archivo CSV y realizar operaciones de limpieza y consolidación, como la fusión de rostros similares o la eliminación de detecciones irrelevantes.
### Resumen de Funciones
**check_create_project_paths(project_name)**: 
Crea la estructura de directorios necesaria para un nuevo proyecto, incluyendo carpetas para datos, rostros detectados y resultados.

**store_faces(video_files, tolerance, project_name, batch_size)**: 
La función principal que procesa un video. Detecta los rostros en cada fotograma, los codifica y los compara con los rostros ya almacenados. Si se encuentra un nuevo rostro, lo guarda como una imagen y registra su información en un archivo CSV.

**delfaceframes(project_name, video_name, corrections)**: 
Permite eliminar detecciones de rostros en fotogramas específicos de un video, lo que es útil para corregir errores de detección o limpiar datos.

**merge_faces_with_gui(project_path, csv)**: 
Proporciona una interfaz gráfica de usuario para fusionar la información de rostros que han sido detectados como diferentes pero que, en realidad, pertenecen a la misma persona. Combina sus datos y elimina las imágenes duplicadas.

**merge_faces(project_path, ids_to_merge)**: 
Similar a la función anterior, pero realiza la fusión de rostros de forma masiva, tomando una lista de IDs de rostros para unirlos.

**delete_faces_with_gui(project_path, csv)**: 
Abre una interfaz de usuario para que puedas seleccionar y eliminar fácilmente las imágenes de rostros irrelevantes de la base de datos y la carpeta de rostros.

**delete_faces_by_ids(project_path, ids_to_delete, csv)**: 
Elimina rostros basándose en una lista de IDs proporcionada en el código, ideal para operaciones de limpieza a gran escala.

**get_faces_data(project_path, csv_filename)**: 
Carga y muestra el contenido del archivo CSV que almacena los datos de los rostros detectados.

## Face_recog-printfunctions.py 🎥
Este script de Python se encarga de superponer información (nombres, alias, datos adicionales) sobre los rostros detectados en un vídeo. Utiliza los datos de un archivo CSV que contiene la información de los rostros, sus posiciones en cada fotograma y otros detalles relevantes. El resultado son imágenes PNG con la superposición de datos, listas para ser integradas en el vídeo original.

### Resumen de Funciones
**adapt_written_bbdd(csv_path)**: Lee un archivo CSV, filtra las filas que contienen datos y elimina columnas irrelevantes para preparar la base de datos de rostros.

**generate_filled_copy(project, csvtogetfrom, newname, csvtotransform)**: Combina los datos de una base de datos de rostros (por ejemplo, con nombres y otra información) con un archivo CSV de posiciones de rostros. Esto crea una nueva base de datos unificada que se usa para generar las superposiciones.

**test_overlay(video_name, frame_number, style_params, faces_data, project)**: Función de prueba que genera y muestra una superposición en un solo fotograma de un vídeo. Es útil para verificar la configuración de estilo (color de caja, tipo de letra, tamaño, etc.) antes de procesar todo el vídeo.

**generate_overlays(video_file, style_params, faces_data_path, project)**: La función principal que recorre todos los fotogramas de un vídeo. Para cada fotograma, lee los datos de posición de los rostros, dibuja un rectángulo alrededor de cada uno y añade un cuadro de texto con la información correspondiente (ID, nombre, etc.). El resultado es una serie de imágenes PNG transparentes (overlays) que se guardan en una carpeta de resultados para su posterior uso.

## Librerías necesarias
Para ejecutar este script, se requieren las siguientes librerías de Python:

+  **face_recognition**: 
Para la detección y codificación de rostros.

+  **tkinter**: 
Para crear interfaces de usuario de selección de archivos.

+  **pandas**: 
Para la manipulación de datos tabulares, como los archivos CSV.

+  **cv2 (OpenCV)**: 
Para leer y procesar los fotogramas del vídeo.

+  **PIL (Pillow)**: 
Para dibujar rectángulos y texto sobre los fotogramas del vídeo.

+  **os**: 
Para gestionar rutas de archivos y directorios.

+  **tqdm**: 
Para mostrar una barra de progreso durante la generación de las superposiciones.

+  **ast**: 
Para convertir cadenas de texto a estructuras de datos de Python.

+  **numpy**: Para operaciones numéricas en los datos de los fotogramas.

