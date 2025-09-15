import os
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import ast
from Face_recog_detectfunctions import check_create_project_paths
import numpy as np

pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
pd.set_option('display.width', 1000)  # Ajusta el ancho de la pantalla para evitar que se recorte
pd.set_option('display.max_colwidth', 50)

def adapt_written_bbdd(csv_path):
    db=pd.read_csv(csv_path)
    print(db.head())
    db=db[(db['Nombre'].notna())]
    columnstodrop = ["id", "appearance", "frame_positions", "face_encoding"]
    db = db.drop(columns=columnstodrop)
    return db


def generate_filled_copy(project, csvtogetfrom, newname, csvtotransform="faces_data.csv"):
    project_path = check_create_project_paths(project)
    positions = pd.read_csv(os.path.join(project_path, "data", csvtotransform))
    faces = csvtogetfrom
    print(faces['cumulative'].head())

    faces['cumulative'] = faces['cumulative'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    faces = faces.explode("cumulative")
    faces = faces.drop(faces.columns[0], axis=1)
    faces = faces.rename(columns={'cumulative': 'id'})

    processed_db = positions.merge(faces, on='id', how='left')
    print(processed_db)

    newcsv_path = os.path.join(project_path, "data", newname)
    processed_db.to_csv(newcsv_path, index=False)
    return processed_db


def test_overlay(video_name, frame_number, style_params, faces_data, project):
    # Ruta a la carpeta 'input_clips' en la raíz del proyecto
    input_clips_folder = os.path.join(os.path.dirname(__file__), "input_clips")

    # Construir la ruta completa del archivo de vídeo
    video_file = os.path.join(input_clips_folder, video_name)

    print(f"Intentando abrir el vídeo: {video_file}")

    # Verificar si el archivo de vídeo existe
    if not os.path.exists(video_file):
        print(f"Archivo de vídeo {video_name} no encontrado en {input_clips_folder}. Saliendo.")
        return

    # Abrir el archivo de vídeo usando OpenCV
    video_capture = cv2.VideoCapture(video_file)

    # Verificar si el vídeo se abrió correctamente
    if not video_capture.isOpened():
        print(f"No se pudo abrir el archivo de vídeo: {video_file}")
        return

    # Leer el fotograma específico
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video_capture.read()
    if not ret:
        print("Error al leer el fotograma.")
        return

    # Convertir el fotograma a una imagen PIL para dibujar superposiciones fácilmente
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Extraer el espaciado de línea y de letra del diccionario style_params, con valores por defecto si no se proporcionan
    line_spacing = style_params.get('line_spacing', 20)  # Valor por defecto 20 si no se establece en style_params
    font_size = style_params.get('font_size', 120)  # Obtener el tamaño de la fuente de style_params
    letter_spacing = style_params.get('letter_spacing', 0)  # Valor por defecto 0 si no se establece en style_params

    print(f"Usando tamaño de fuente: {font_size}")
    print(f"Usando espaciado de línea: {line_spacing}")
    print(f"Usando espaciado de letra: {letter_spacing}")

    # Procesar los datos de las caras para el nombre del vídeo especificado
    for index, row in faces_data.iterrows():
        if video_name not in row['frame_positions']:  # Omitir si el vídeo no está en frame_positions
            continue

        # Cargar las posiciones de los fotogramas desde el CSV (como una cadena de lista de tuplas)
        frame_positions = eval(row['frame_positions'])[video_name]

        if frame_number < len(frame_positions):
            position = frame_positions[frame_number]
            if position == "not_seen":
                # print(f"{row['Nombre']} omitido.")
                continue

            top, right, bottom, left = position
            # Dibujar el cuadro delimitador
            draw.rectangle([left, top, right, bottom], outline=style_params['box_color'],
                            width=style_params['stroke_width'])

            # Manejo de la fuente
            try:
                font = ImageFont.truetype(style_params['font_path'], font_size)  # Asegurar que se aplique el tamaño de fuente correcto
            except IOError:
                print("Error con la fuente")
                font = ImageFont.load_default()

            # Definir los textos en el orden deseado, formateados con los nombres de las columnas
            text_lines = [
                f"Id: {row['id']}",
                f"{row['Nombre']}  {f'- {row['AKA']}' if pd.notna(row['AKA']) else ''}",
                f"{row['Data 1']}" if pd.notna(row['Data 1']) else '',
                f"{row['Data 2']}" if pd.notna(row['Data 2']) else ''
            ]

            # Eliminar líneas de texto vacías (las que están vacías por valores NaN)
            text_lines = [line for line in text_lines if line.strip() != '']

            # Función para envolver texto si excede una cierta longitud (40 caracteres)
            def wrap_text(text, max_length=40):
                # Dividir el texto en palabras
                words = text.split()
                lines = []
                current_line = words[0]

                for word in words[1:]:
                    # Si la línea actual + la nueva palabra excede max_length, envolver la línea
                    if len(current_line + ' ' + word) <= max_length:
                        current_line += ' ' + word
                    else:
                        lines.append(current_line)
                        current_line = word

                # Agregar la última línea
                lines.append(current_line)
                return lines

            # Calcular la altura total del bloque de texto
            total_lines = 0  # Para llevar un registro de cuántas líneas de texto se dibujarán
            for text in text_lines:
                wrapped_lines = wrap_text(text, max_length=22)
                total_lines += len(wrapped_lines)

            # Altura del bloque de texto (basado en el tamaño de la fuente y el espaciado de línea)
            total_text_height = total_lines * font_size + (total_lines - 1) * line_spacing

            # Calcular la posición Y inicial del texto para evitar la superposición con el cuadro delimitador
            text_y_position = top - total_text_height - 20  # 10px para un pequeño margen

            # Función para dibujar texto con espaciado de letra
            def draw_text_with_spacing(draw, xy, text, font, fill, spacing):
                x, y = xy
                for char in text:
                    draw.text((x, y), char, font=font, fill=fill)
                    x += font.getbbox(char)[2] + spacing

            # Dibujar cada línea envuelta
            for text in text_lines:
                wrapped_lines = wrap_text(text, max_length=22)

                for line in wrapped_lines:
                    draw_text_with_spacing(draw, (left, text_y_position), line, font, style_params['text_color'], letter_spacing)
                    text_y_position += font_size + line_spacing  # Mover la posición Y hacia abajo después de cada línea

    # Guardar la imagen de superposición de prueba
    test_overlay_path = os.path.join(os.path.dirname(__file__), "projects", project, "test", f"{video_name}_frame_{frame_number:04d}.png")
    pil_image.show()  # Mostrar la imagen (puedes eliminar esto si no lo necesitas)

    # Opcionalmente, guardar la imagen
    pil_image.save(test_overlay_path)

    print(f"Superposición de prueba guardada en {test_overlay_path}")


def generate_overlays(video_file, style_params, faces_data_path, project):
    # Cargar datos del CSV
    faces_data = pd.read_csv(faces_data_path)
    video_path = os.path.join("input_clips", video_file)
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_folder = os.path.join("projects", project, "results", os.path.splitext(os.path.basename(video_file))[0])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Asegurarse de que los colores sean tuplas de cuatro elementos (RGBA)
    box_color = style_params['box_color']
    if len(box_color) == 3:
        box_color = (*box_color, 255)  # Agregar opacidad total si no está presente

    text_color = style_params['text_color']
    if len(text_color) == 3:
        text_color = (*text_color, 255)  # Agregar opacidad total si no está presente

    for frame_number in tqdm(range(frame_count), desc=f"Generando superposición para {video_file}"):
        ret, frame = video_capture.read()
        if not ret:
            break

        # Crear una superposición transparente
        overlay = Image.new("RGBA", (frame.shape[1], frame.shape[0]), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Verificar si se detectan caras
        face_detected = False

        for index, row in faces_data.iterrows():
            if video_file not in row['frame_positions']:  # Omitir si el vídeo no está en frame_positions
                continue

            # Iterar sobre las posiciones de los fotogramas para este vídeo
            frame_positions = eval(row['frame_positions'])[video_file]
            if frame_number < len(frame_positions):
                position = frame_positions[frame_number]
                if position == "not_seen":
                    continue
                face_detected = True
                top, right, bottom, left = position
                # Dibujar el cuadro delimitador
                draw.rectangle([left, top, right, bottom], outline=box_color,
                                width=style_params['stroke_width'])

                # Manejo de la fuente
                try:
                    font = ImageFont.truetype(style_params['font_path'], style_params['font_size'])
                except IOError:
                    font = ImageFont.load_default()

                # Definir los textos en el orden deseado, formateados con los nombres de las columnas
                text_lines = [
                    f"Id: {row['id']}",
                    f"{row['Nombre']}  {f'- {row['AKA']}' if pd.notna(row['AKA']) else ''}",
                    f"{row['Data 1']}" if pd.notna(row['Data 1']) else '',
                    f"{row['Data 2']}" if pd.notna(row['Data 2']) else ''
                ]

                # Eliminar líneas de texto vacías (las que están vacías por valores NaN)
                text_lines = [line for line in text_lines if line.strip() != '']

                # Función para envolver texto si excede una cierta longitud (40 caracteres)
                def wrap_text(text, max_length=40):
                    # Dividir el texto en palabras
                    words = text.split()
                    lines = []
                    current_line = words[0]

                    for word in words[1:]:
                        # Si la línea actual + la nueva palabra excede max_length, envolver la línea
                        if len(current_line + ' ' + word) <= max_length:
                            current_line += ' ' + word
                        else:
                            lines.append(current_line)
                            current_line = word

                    # Agregar la última línea
                    lines.append(current_line)
                    return lines

                # Calcular la altura total del bloque de texto
                total_lines = 0  # Para llevar un registro de cuántas líneas de texto se dibujarán
                for text in text_lines:
                    wrapped_lines = wrap_text(text, max_length=22)
                    total_lines += len(wrapped_lines)

                # Altura del bloque de texto (basado en el tamaño de la fuente y el espaciado de línea)
                total_text_height = total_lines * style_params['font_size'] + (total_lines - 1) * style_params.get(
                    'line_spacing', 20)

                # Calcular la posición Y inicial del texto para evitar la superposición con el cuadro delimitador
                text_y_position = top - total_text_height - 20  # 10px para un pequeño margen

                # Función para dibujar texto con espaciado de letra
                def draw_text_with_spacing(draw, xy, text, font, fill, spacing):
                    x, y = xy
                    for char in text:
                        draw.text((x, y), char, font=font, fill=fill)
                        x += font.getbbox(char)[2] + spacing

                # Dibujar cada línea envuelta
                for text in text_lines:
                    wrapped_lines = wrap_text(text, max_length=22)

                    for line in wrapped_lines:
                        draw_text_with_spacing(draw, (left, text_y_position), line, font, text_color,
                                               style_params.get('letter_spacing', 0))
                        text_y_position += style_params['font_size'] + style_params.get('line_spacing',
                                                                                       20)  # Mover la posición Y hacia abajo después de cada línea

        # Guardar la superposición
        output_path = os.path.join(output_folder, f"{frame_number:04d}.png")
        overlay.save(output_path, "PNG")

    print(f"Superposiciones generadas y guardadas en {output_folder}")

print("Librerías cargadas del chill")



project = "project3-tol0.5 - mergeos y borrados"
csvtogetfrom = pd.read_csv(r"C:\...\PythonProject\written_ddbbs\bbdd_filled1.csv")
generate_filled_copy(project, csvtogetfrom, "procesado.csv")

#
style_params = {
    'box_color': (255, 255, 255),
    'stroke_width': 3,
    'text_color': (255, 255, 255),
    'font_path': r'C:\...\....ttf',  # Asegúrate de tener la ruta correcta al archivo de la fuente
    'font_size': 15,  # Tamaño de fuente aumentado para vídeo 2K
    'line_spacing': 2,
    'letter_spacing': 2
}


faces_data = r"C:\...\PythonProject\projects\project3-tol0.5 - mergeos y borrados\data\procesado.csv"
video_file = "Filtered_Facetracking.mp4"

generate_overlays(video_file, style_params,faces_data, project)
