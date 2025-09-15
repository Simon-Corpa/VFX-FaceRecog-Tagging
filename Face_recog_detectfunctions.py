import os
import cv2
import pandas as pd
from tqdm import tqdm
import face_recognition
import numpy as np
from tkinter import filedialog, Tk


# Ajustes de visualización para pandas
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', 1000)  
pd.set_option('display.max_colwidth', 50)

project_path = ""

# Crear o abrir carpeta de proyecto
def check_create_project_paths(project_name):
    """
    Comprueba la estructura que la estructura del proyecto fijada para ejecutar la tarea existe y, en su defecto, la crea.
    """
    global project_path
    project_path = os.path.join("projects", project_name)
    faces_dir = os.path.join(project_path, "data", "faces")  
    data_dir = os.path.join(project_path, "data")
    results_dir = os.path.join(project_path, "results")
    test_dir = os.path.join(project_path, "test")

    # Crear la estructura de carpetas si no existe
    if not os.path.exists(project_path):
        os.makedirs(project_path)
        os.makedirs(data_dir)
        os.makedirs(faces_dir)
        os.makedirs(results_dir)
        os.makedirs(test_dir)

    print(f"Proyecto '{project_name}' cargado.")
    return project_path


# Almacenar las caras detectadas en el input de video en el directorio faces_dir
def store_faces(video_files, tolerance, project_name="your_project_name", batch_size=32):
    """
    Procesa el vídeo de input, crea el proyecto de este si fuese necesario, detecta las caras con la lectura simultanea de frames en batches.
    Tras ello guarda la información de la posición de las caras en un dataset y reserva las imagenes de las caras en una carpeta para su posterior identificación.
    """
    project_path = check_create_project_paths(project_name)

    # Definición de rutas y comprobación de existencia
    data_dir = os.path.join(project_path, "data") 
    faces_dir = os.path.join(data_dir, "faces") 
    csv_path = os.path.join(data_dir, "faces_data.csv")
    os.makedirs(faces_dir, exist_ok=True)

    # Comprobar existencia o crear un csv para el almacenamiento de las caras y sus posiciones en pantalla
    if os.path.exists(csv_path):
        faces_data = pd.read_csv(csv_path, dtype={'frame_positions': str})  
        faces_data['frame_positions'] = faces_data['frame_positions'].apply(eval)
    else:
        faces_data = pd.DataFrame(columns=["id", "appearance", "frame_positions", "face_encoding"])

    # Fijar el directorio para los vídeos
    input_clips_dir = os.path.join(os.getcwd(), "input_clips")  # Correct path to 'input_clips' folder

    # Procesar el video de input
    for video_file in video_files:
        video_path = os.path.join(input_clips_dir, video_file)
        print(f"Processing video: {video_path}")

        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            continue

        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {frame_count}")
        if frame_count == 0:
            print(f"Error: No frames detected in {video_path}")
            continue

        face_id_counter = len(faces_data)

        # Iterar entre frames en batches
        total_batches = frame_count // batch_size + int(frame_count % batch_size != 0)
        for batch_idx in tqdm(range(total_batches), desc=f"Processing batches of {video_file}"):
            frames = []
            valid_frame_indices = []

            # Cargar un batch para el procesado
            for _ in tqdm(range(batch_size), desc=f"Loading frames for batch {batch_idx + 1}/{total_batches}", leave=False):
                ret, frame = video_capture.read()
                if not ret:
                    break 

                # Comprobar si el fotograma está en negro
                if not np.any(frame):
                    for index, row in faces_data.iterrows():
                        frame_positions = eval(row["frame_positions"])
                        if video_file not in frame_positions:
                            frame_positions[video_file] = ["not_seen"] * frame_count
                        frame_positions[video_file].append("not_seen")
                        faces_data.at[index, "frame_positions"] = str(frame_positions)
                else:
                    frames.append(frame)
                    valid_frame_indices.append(batch_idx * batch_size + len(frames))

            # Saltar el proceso si no hay fotogramas validos 
            if not frames:
                continue

            # Detectar las caras de cada batch
            face_locations_batch = []
            face_encodings_batch = []
            for frame_idx, frame in enumerate(tqdm(frames, desc=f"Processing frames in batch {batch_idx + 1}/{total_batches}", leave=False)):
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                face_locations_batch.append(face_locations)
                face_encodings_batch.append(face_encodings)

            # Genera una lista con la información del frame y la localización de la cara y comprueba la existencia de esa cara en la carpeta de caras
            for frame_idx, (face_locations, face_encodings) in enumerate(zip(face_locations_batch, face_encodings_batch)):
                detected_face_ids = []
                frame_number = valid_frame_indices[frame_idx]

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # Convierte a lista la información
                    face_encoding_list = face_encoding.tolist()

                    # Comprueba si esta cara aparece previamente
                    match = None
                    for index, row in faces_data.iterrows():
                        stored_encoding = np.array(eval(row['face_encoding']))
                        if face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=tolerance)[0]:
                            match = index
                            break

                    if match is None:
                        # Nueva cara
                        new_id = f"{face_id_counter:05d}"
                        face_image_path = os.path.join(faces_dir, f"{new_id}.jpg")

                        # Guardar la cara
                        top, right, bottom, left = face_location
                        top = max(0, top - 20)
                        right = min(frame.shape[1], right + 20)
                        bottom = min(frame.shape[0], bottom + 20)
                        left = max(0, left - 20)

                        face_image = frames[frame_idx][top:bottom, left:right]
                        cv2.imwrite(face_image_path, face_image)

                        frame_positions = {video_file: ["not_seen"] * (frame_number) + [face_location]}

                        new_data = pd.DataFrame([{
                            "id": new_id,
                            "appearance": 1,  # Starts with 1 appearance
                            "frame_positions": str(frame_positions),  # Store as string for later parsing
                            "face_encoding": str(face_encoding_list)  # Save face encoding as a string
                        }])

                        faces_data = pd.concat([faces_data, new_data], ignore_index=True)
                        face_id_counter += 1

                        print(f"New face recognized: ID {new_id} detected in frame {frame_number} of {video_file}.")
                    else:
                        # Actualiza la cara existente y guarda la nueva información de posición
                        detected_face_ids.append(match)
                        frame_positions = eval(faces_data.at[match, "frame_positions"])
                        if video_file not in frame_positions:
                            frame_positions[video_file] = ["not_seen"] * frame_count
                        if len(frame_positions[video_file]) < frame_count:
                            frame_positions[video_file].extend(["not_seen"] * (frame_count - len(frame_positions[video_file])))
                        frame_positions[video_file][frame_number] = face_location
                        faces_data.at[match, "frame_positions"] = str(frame_positions)
                        faces_data.at[match, "appearance"] += 1

                # Guardar en el resto de caras información sobre ese fotograma vacío
                for index, row in faces_data.iterrows():
                    if index not in detected_face_ids:
                        frame_positions = eval(row["frame_positions"])
                        if video_file not in frame_positions:
                            frame_positions[video_file] = ["not_seen"] * frame_count
                        if len(frame_positions[video_file]) < frame_count:
                            frame_positions[video_file].extend(["not_seen"] * (frame_count - len(frame_positions[video_file])))
                        frame_positions[video_file][frame_number] = "not_seen"
                        faces_data.at[index, "frame_positions"] = str(frame_positions)

            # Guardar la tabla de caras en csv
            faces_data = faces_data.sort_values(by="appearance", ascending=False)
            faces_data.to_csv(csv_path, index=False)

    print("Face storage completed.")



# Function to delete frames and update the database with 'not_seen'
def delfaceframes(project_name, video_name, corrections):
    """
    Función para eliminar detecciones de fotogramas concretos del vídeo.
    Ejp: Fotogramas borrosos, que no salen caras, negros, etc.
    """
    project_path = os.path.join("projects", project_name)
    csv_path = os.path.join(project_path, "data", "faces_data.csv")

    # Check if faces data CSV exists
    if not os.path.exists(csv_path):
        print(f"No faces data found for project '{project_name}'. Exiting.")
        return

    # Load the faces data
    faces_data = pd.read_csv(csv_path)

    if faces_data.empty:
        print(f"Faces data is empty for project '{project_name}'. Exiting.")
        return

    # Iterate over the corrections dictionary
    for face_id, frames_to_modify in corrections.items():
        # Ensure the frames_to_modify is a list
        if isinstance(frames_to_modify, int):
            frames_to_modify = [frames_to_modify]

        # Locate the face row
        face_row = faces_data.loc[faces_data["id"] == face_id]
        if face_row.empty:
            print(f"Face ID {face_id} not found in the database. Skipping.")
            continue

        face_index = face_row.index[0]
        frame_positions = eval(faces_data.at[face_index, "frame_positions"])

        if video_name not in frame_positions:
            print(f"Video '{video_name}' not found for Face ID {face_id}. Skipping.")
            continue

        # Replace the specified frames with "not_seen"
        for frame in frames_to_modify:
            if frame < len(frame_positions[video_name]):
                frame_positions[video_name][frame] = "not_seen"
            else:
                print(f"Frame {frame} out of range for Face ID {face_id} in video '{video_name}'. Skipping.")

        faces_data.at[face_index, "frame_positions"] = str(frame_positions)

    # Guardar el nuevo dataset
    faces_data.to_csv(csv_path, index=False)
    print(f"Database updated for project '{project_name}'. Frames replaced with 'not_seen' for video '{video_name}'.")

# Combinar caras existentes con una interfaz si se interpretasen como distintas
def merge_faces_with_gui(project_path, csv="faces_data.csv"):
    """
    Fusionar, con la interfaz de selección de archivos, la información de caras iguales  de archivos que han sido detectadas como distintas dada la tolerancia prefijada durante la detección.
    Selecciona las distintas caras guardadas para fusionarlas y adaptar la base de datos y la carpeta con caras guardadas.
    """
    import os
    import pandas as pd
    from tkinter import Tk, filedialog

    csv_path = os.path.join(project_path, "data", csv)
    if not os.path.exists(csv_path):
        print(f"No faces data found at {csv_path}. Exiting.")
        return

    # Cargar caras
    faces_data = pd.read_csv(csv_path)
    faces_data['id'] = faces_data['id'].astype(str)

    # Comprobar existencia de columnas
    if "cumulative" not in faces_data.columns:
        faces_data["cumulative"] = None
    if "merged_ids" not in faces_data.columns:
        faces_data["merged_ids"] = None

    while True:
        # Abrir Tkinter para seleccionar varios archivos
        Tk().withdraw()
        file_paths = filedialog.askopenfilenames(
            title="Select images to merge",
            initialdir=os.path.join(project_path, "data", "faces"),
            filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
        )
        if not file_paths:
            print("No files selected. Exiting the process.")
            break

        # Extraer los IDs de las imagenes seleccionadas
        ids_to_merge = [os.path.splitext(os.path.basename(file_path))[0].lstrip('0') for file_path in file_paths]
        ids_to_merge = sorted(ids_to_merge)  # Sort IDs to ensure the smallest is first

        print(f"Selected IDs to merge: {ids_to_merge}")

     
        rows_to_merge = faces_data[faces_data['id'].isin(ids_to_merge)]
        if rows_to_merge.empty:
            print(f"No matching IDs found for: {ids_to_merge}. Skipping.")
            continue
        
        #Preparamos la información de la nueva linea
        new_id = ids_to_merge[0]
        new_face_image_path = os.path.join(project_path, "data", "faces", f"{new_id}.jpg")
        new_frame_positions = {}
        merged_ids = []
        new_cumulative = []

        # Iterar por cada linea para actualizar la información
        for _, row in rows_to_merge.iterrows():
            merged_ids.extend(eval(row.get('merged_ids', '[]')))
            merged_ids.append(row['id'])

            # Actualizar la posición en cada fotograma
            frame_positions = eval(row['frame_positions'])
            for video_file, positions in frame_positions.items():
                if video_file not in new_frame_positions:
                    new_frame_positions[video_file] = []

                # Add only unique positions
                for pos in positions:
                    if pos not in new_frame_positions[video_file]:
                        new_frame_positions[video_file].append(pos)

            # Actualizar la columna acumualada
            if pd.notna(row["cumulative"]):  
                cumulative_values = eval(row["cumulative"])
                new_cumulative.extend(cumulative_values)

        merged_ids = list(set(merged_ids))
        new_cumulative = sorted(set(new_cumulative))
        new_appearance = rows_to_merge['appearance'].sum()

        # Actualizar el dataset con la nueva información
        faces_data.loc[faces_data['id'] == new_id, 'frame_positions'] = str(new_frame_positions)
        faces_data.loc[faces_data['id'] == new_id, 'appearance'] = new_appearance
        faces_data.loc[faces_data['id'] == new_id, 'merged_ids'] = str(merged_ids)
        faces_data.loc[faces_data['id'] == new_id, 'cumulative'] = str(new_cumulative)

        # Eliminar las imagenes redundantes de la cara
        for id_to_delete in ids_to_merge[1:]:
            id_to_delete = id_to_delete.zfill(5)
            image_path_to_delete = os.path.join(project_path, "data", "faces", f"{id_to_delete}.jpg")
            if os.path.exists(image_path_to_delete):
                os.remove(image_path_to_delete)
                print(f"Deleted image: {image_path_to_delete}")
        faces_data = faces_data[~faces_data['id'].isin(ids_to_merge[1:])]
        print(f"Successfully merged IDs: {ids_to_merge}")

    # Guardar el nuevo dataset
    faces_data.to_csv(csv_path, index=False)

    print("All faces merged, images deleted, and data saved successfully.")


def merge_faces(project_path, ids_to_merge):
    """
    Mergeamos las caras con una lista de IDs (títulos de las fotos de las caras) o con una lista de listas de IDs.
    Más cómodo para hacer el proceso de forma masiva.
    """
    # Comprobar si es una lista de elementos o de listas
    if isinstance(ids_to_merge[0], list):
        merge_groups = ids_to_merge
    else:
        merge_groups = [ids_to_merge]

    csv_path = os.path.join(project_path, "data", "faces_data.csv")
    if not os.path.exists(csv_path):
        print(f"No faces data found at {csv_path}. Exiting.")
        return

    # Cargar la data de las caras
    faces_data = pd.read_csv(csv_path)

    if "cumulative" not in faces_data.columns:
        faces_data["cumulative"] = faces_data["id"].apply(lambda x: [x])

    for ids in merge_groups:
        ids = sorted(ids)  # Ordenar para que el ID más pequeño sea el primero

        # Fetch the rows for the given IDs
        rows_to_merge = faces_data[faces_data['id'].isin(ids)]

        new_id = ids[0]
        new_frame_positions = {}
        new_cumulative = []

        # Iterar por cada fila y actualizar data
        for _, row in rows_to_merge.iterrows():
            frame_positions = eval(row['frame_positions'])
            for video_file, positions in frame_positions.items():
                if video_file not in new_frame_positions:
                    new_frame_positions[video_file] = []
                for pos in positions:
                    if pos not in new_frame_positions[video_file]:
                        new_frame_positions[video_file].append(pos)
        
            cumulative_value = row["cumulative"]
            if isinstance(cumulative_value, str) and cumulative_value != "[]" and pd.notna(cumulative_value):
                try:
                    cumulative_values = eval(cumulative_value) 
                    if isinstance(cumulative_values, list):
                        new_cumulative.extend(cumulative_values)
                    else:
                        print(f"Invalid cumulative format in row with id {row['id']}")
                except Exception as e:
                    print(f"Error processing cumulative data: {e}")
            elif isinstance(cumulative_value, list) and len(cumulative_value) > 0:
                new_cumulative.extend(cumulative_value)  

        new_appearance = rows_to_merge['appearance'].sum()
        new_cumulative = sorted(set(new_cumulative))

        # Actualiza cada nueva fila en el dataset
        faces_data.loc[faces_data['id'] == new_id, 'frame_positions'] = str(new_frame_positions)
        faces_data.loc[faces_data['id'] == new_id, 'appearance'] = new_appearance
        faces_data.loc[faces_data['id'] == new_id, 'cumulative'] = str(new_cumulative)

        # Eliminar las imagenes redundantes de la cara
        for id_to_delete in ids[1:]:
            id_to_delete_str = str(id_to_delete).zfill(5)  # Convert to string and pad to 5 digits
            image_path_to_delete = os.path.join(project_path, "data", "faces", f"{id_to_delete_str}.jpg")

            print(f"Attempting to delete image at: {image_path_to_delete}")

            if os.path.exists(image_path_to_delete):
                try:
                    os.remove(image_path_to_delete)
                    print(f"Deleted image: {image_path_to_delete}")
                except Exception as e:
                    print(f"Error deleting image {image_path_to_delete}: {e}")
            else:
                print(f"Image not found: {image_path_to_delete}")

        # Eliminar las filas de IDs redundantes 
        faces_data = faces_data[~faces_data['id'].isin(ids[1:])]

    # Guardamos la nueva información de las caras a un dataset
    faces_data = faces_data.sort_values(by="appearance", ascending=False)
    faces_data.to_csv(csv_path, index=False)

    print("Faces merged, images deleted, and data saved successfully.")



# Borrado de caras con interfaz
def delete_faces_with_gui(project_path, csv="faces_data.csv"):
    """
    Eliminar caras irrelevantes mediante la selección de sus imagenes en la carpeta de caras con el gestor de archivos.
    """

    csv_path = os.path.join(project_path, "data", csv)
    if not os.path.exists(csv_path):
        print(f"No faces data found at {csv_path}. Exiting.")
        return
    faces_data = pd.read_csv(csv_path)
    faces_data['id'] = faces_data['id'].astype(str)

    while True:
        # Abre Tkinter para seleccionar las caras a borrar
        Tk().withdraw()
        file_paths = filedialog.askopenfilenames(
            title="Select images to delete",
            initialdir=os.path.join(project_path, "data", "faces"),
            filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
        )
        if not file_paths:
            print("No files selected. Exiting the process.")
            break

        # Obten los IDs de los archivos señalados
        ids_to_delete = [os.path.splitext(os.path.basename(file_path))[0].lstrip('0') for file_path in file_paths]
        ids_to_delete = sorted(ids_to_delete)  # Sort IDs to ensure the smallest is first

        print(f"Selected IDs to delete: {ids_to_delete}")

        rows_to_delete = faces_data[faces_data['id'].isin(ids_to_delete)]
        if rows_to_delete.empty:
            print(f"No matching IDs found for: {ids_to_delete}. Skipping.")
            continue

        # Elimina las imagenes
        for id_to_delete in ids_to_delete:
            id_to_delete = id_to_delete.zfill(5)
            image_path_to_delete = os.path.join(project_path, "data", "faces", f"{id_to_delete}.jpg")
            if os.path.exists(image_path_to_delete):
                os.remove(image_path_to_delete)
                print(f"Deleted image: {image_path_to_delete}")

        # Elimina las filas
        faces_data = faces_data[~faces_data['id'].isin(ids_to_delete)]

        print(f"Successfully deleted IDs: {ids_to_delete}")

    # Guarda el dataset
    faces_data.to_csv(csv_path, index=False)

    print("All selected faces deleted, images removed, and data saved successfully.")

def delete_faces_by_ids(project_path, ids_to_delete, csv="faces_data.csv"):
    """
    Eliminar caras irrelevantes con una lista de IDs de esas caras.
    """

    csv_path = os.path.join(project_path, "data", csv)
    if not os.path.exists(csv_path):
        print(f"No faces data found at {csv_path}. Exiting.")
        return
    faces_data = pd.read_csv(csv_path)

    faces_data['id'] = faces_data['id'].astype(str)
    ids_to_delete = [str(id_to_delete).lstrip('0') for id_to_delete in ids_to_delete]
    ids_to_delete = sorted(ids_to_delete)  # Sort IDs for consistency

    print(f"IDs to delete: {ids_to_delete}")

    rows_to_delete = faces_data[faces_data['id'].isin(ids_to_delete)]

    if rows_to_delete.empty:
        print(f"No matching IDs found for: {ids_to_delete}. Exiting.")
        return

    # Elimina las imagenes
    for id_to_delete in ids_to_delete:
        id_to_delete = id_to_delete.zfill(5)  
        image_path_to_delete = os.path.join(project_path, "data", "faces", f"{id_to_delete}.jpg")
        if os.path.exists(image_path_to_delete):
            os.remove(image_path_to_delete)
            print(f"Deleted image: {image_path_to_delete}")
        else:
            print(f"Image not found for ID {id_to_delete}, skipping.")

    # Elimina las filas
    faces_data = faces_data[~faces_data['id'].isin(ids_to_delete)]

    # Guarda el dataset
    faces_data.to_csv(csv_path, index=False)

    print(f"Successfully deleted IDs: {ids_to_delete}. Data saved back to {csv_path}.")

def get_faces_data(project_path, csv_filename="faces_data.csv"):
    """
    Mostrar el dataset de las caras.
    """
    csv_path = os.path.join(project_path, "data", csv_filename)

    if not os.path.exists(csv_path):
        print(f"No faces data found at {csv_path}. Exiting.")
        return None

    faces_data = pd.read_csv(csv_path)
    print(f"Faces data loaded from {csv_path}.")

    return faces_data
