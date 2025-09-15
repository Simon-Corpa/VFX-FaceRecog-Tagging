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
    # Path to the 'input_clips' folder in the root of your project
    input_clips_folder = os.path.join(os.path.dirname(__file__), "input_clips")

    # Construct the full path of the video file
    video_file = os.path.join(input_clips_folder, video_name)

    print(f"Attempting to open video: {video_file}")

    # Check if the video file exists
    if not os.path.exists(video_file):
        print(f"Video file {video_name} not found in {input_clips_folder}. Exiting.")
        return

    # Open the video file using OpenCV
    video_capture = cv2.VideoCapture(video_file)

    # Check if video opened successfully
    if not video_capture.isOpened():
        print(f"Failed to open video file: {video_file}")
        return

    # Read the specific frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video_capture.read()
    if not ret:
        print("Error reading frame.")
        return

    # Convert frame to PIL image for easier overlay drawing
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Extract line spacing and letter spacing from the style_params dictionary, with default values if not provided
    line_spacing = style_params.get('line_spacing', 20)  # Default to 20 if not set in style_params
    font_size = style_params.get('font_size', 120)  # Get font size from style_params
    letter_spacing = style_params.get('letter_spacing', 0)  # Default to 0 if not set in style_params

    print(f"Using font size: {font_size}")
    print(f"Using line spacing: {line_spacing}")
    print(f"Using letter spacing: {letter_spacing}")

    # Process faces data for the specified video name
    for index, row in faces_data.iterrows():
        if video_name not in row['frame_positions']:  # Skip if video is not in frame_positions
            continue

        # Load the frame positions from the CSV (as string of list of tuples)
        frame_positions = eval(row['frame_positions'])[video_name]

        if frame_number < len(frame_positions):
            position = frame_positions[frame_number]
            if position == "not_seen":
                # print(f"{row['Nombre']} skipped.")
                continue

            top, right, bottom, left = position
            # Draw the bounding box
            draw.rectangle([left, top, right, bottom], outline=style_params['box_color'],
                           width=style_params['stroke_width'])

            # Font handling
            try:
                font = ImageFont.truetype(style_params['font_path'], font_size)  # Ensure correct font size is applied
            except IOError:
                print("Error with the font")
                font = ImageFont.load_default()

            # Define the texts in the order you want, formatted with column names
            text_lines = [
                f"Id: {row['id']}",
                f"{row['Nombre']}  {f'- {row['AKA']}' if pd.notna(row['AKA']) else ''}",
                f"{row['Data 1']}" if pd.notna(row['Data 1']) else '',
                f"{row['Data 2']}" if pd.notna(row['Data 2']) else ''
            ]

            # Remove empty text lines (those that are empty because of NaN values)
            text_lines = [line for line in text_lines if line.strip() != '']

            # Function to wrap text if it exceeds a certain length (40 characters)
            def wrap_text(text, max_length=40):
                # Split the text into words
                words = text.split()
                lines = []
                current_line = words[0]

                for word in words[1:]:
                    # If the current line + new word exceeds max_length, wrap the line
                    if len(current_line + ' ' + word) <= max_length:
                        current_line += ' ' + word
                    else:
                        lines.append(current_line)
                        current_line = word

                # Add the last line
                lines.append(current_line)
                return lines

            # Calculate the total height of the text block
            total_lines = 0  # To keep track of how many lines of text will be drawn
            for text in text_lines:
                wrapped_lines = wrap_text(text, max_length=22)
                total_lines += len(wrapped_lines)

            # Height of the text block (based on font size and line spacing)
            total_text_height = total_lines * font_size + (total_lines - 1) * line_spacing

            # Calculate the initial text Y-position to avoid overlap with the bounding box
            text_y_position = top - total_text_height - 20  # 10px for a small margin

            # Function to draw text with letter spacing
            def draw_text_with_spacing(draw, xy, text, font, fill, spacing):
                x, y = xy
                for char in text:
                    draw.text((x, y), char, font=font, fill=fill)
                    x += font.getbbox(char)[2] + spacing

            # Draw each wrapped line
            for text in text_lines:
                wrapped_lines = wrap_text(text, max_length=22)

                for line in wrapped_lines:
                    draw_text_with_spacing(draw, (left, text_y_position), line, font, style_params['text_color'], letter_spacing)
                    text_y_position += font_size + line_spacing  # Move the Y position down after each line

    # Save the test overlay image
    test_overlay_path = os.path.join(os.path.dirname(__file__), "projects", project, "test", f"{video_name}_frame_{frame_number:04d}.png")
    pil_image.show()  # Display the image (you can remove this if you don't need it)

    # Optionally save the image
    pil_image.save(test_overlay_path)

    print(f"Test overlay saved to {test_overlay_path}")


def generate_overlays(video_file, style_params, faces_data_path, project):
    # Load CSV data
    faces_data = pd.read_csv(faces_data_path)
    video_path = os.path.join("input_clips", video_file)
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_folder = os.path.join("projects", project, "results", os.path.splitext(os.path.basename(video_file))[0])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ensure colors are tuples of four elements (RGBA)
    box_color = style_params['box_color']
    if len(box_color) == 3:
        box_color = (*box_color, 255)  # Add full opacity if not present

    text_color = style_params['text_color']
    if len(text_color) == 3:
        text_color = (*text_color, 255)  # Add full opacity if not present

    for frame_number in tqdm(range(frame_count), desc=f"Generating overlay for {video_file}"):
        ret, frame = video_capture.read()
        if not ret:
            break

        # Create a transparent overlay
        overlay = Image.new("RGBA", (frame.shape[1], frame.shape[0]), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Check if no faces are detected
        face_detected = False

        for index, row in faces_data.iterrows():
            if video_file not in row['frame_positions']:  # Skip if video is not in frame_positions
                continue

            # Iterate over frame positions for this video
            frame_positions = eval(row['frame_positions'])[video_file]
            if frame_number < len(frame_positions):
                position = frame_positions[frame_number]
                if position == "not_seen":
                    continue
                face_detected = True
                top, right, bottom, left = position
                # Draw the bounding box
                draw.rectangle([left, top, right, bottom], outline=box_color,
                               width=style_params['stroke_width'])

                # Font handling
                try:
                    font = ImageFont.truetype(style_params['font_path'], style_params['font_size'])
                except IOError:
                    font = ImageFont.load_default()

                # Define the texts in the order you want, formatted with column names
                text_lines = [
                    f"Id: {row['id']}",
                    f"{row['Nombre']}  {f'- {row['AKA']}' if pd.notna(row['AKA']) else ''}",
                    f"{row['Data 1']}" if pd.notna(row['Data 1']) else '',
                    f"{row['Data 2']}" if pd.notna(row['Data 2']) else ''
                ]

                # Remove empty text lines (those that are empty because of NaN values)
                text_lines = [line for line in text_lines if line.strip() != '']

                # Function to wrap text if it exceeds a certain length (40 characters)
                def wrap_text(text, max_length=40):
                    # Split the text into words
                    words = text.split()
                    lines = []
                    current_line = words[0]

                    for word in words[1:]:
                        # If the current line + new word exceeds max_length, wrap the line
                        if len(current_line + ' ' + word) <= max_length:
                            current_line += ' ' + word
                        else:
                            lines.append(current_line)
                            current_line = word

                    # Add the last line
                    lines.append(current_line)
                    return lines

                # Calculate the total height of the text block
                total_lines = 0  # To keep track of how many lines of text will be drawn
                for text in text_lines:
                    wrapped_lines = wrap_text(text, max_length=22)
                    total_lines += len(wrapped_lines)

                # Height of the text block (based on font size and line spacing)
                total_text_height = total_lines * style_params['font_size'] + (total_lines - 1) * style_params.get(
                    'line_spacing', 20)

                # Calculate the initial text Y-position to avoid overlap with the bounding box
                text_y_position = top - total_text_height - 20  # 10px for a small margin

                # Function to draw text with letter spacing
                def draw_text_with_spacing(draw, xy, text, font, fill, spacing):
                    x, y = xy
                    for char in text:
                        draw.text((x, y), char, font=font, fill=fill)
                        x += font.getbbox(char)[2] + spacing

                # Draw each wrapped line
                for text in text_lines:
                    wrapped_lines = wrap_text(text, max_length=22)

                    for line in wrapped_lines:
                        draw_text_with_spacing(draw, (left, text_y_position), line, font, text_color,
                                               style_params.get('letter_spacing', 0))
                        text_y_position += style_params['font_size'] + style_params.get('line_spacing',
                                                                                        20)  # Move the Y position down after each line

        # Save the overlay
        output_path = os.path.join(output_folder, f"{frame_number:04d}.png")
        overlay.save(output_path, "PNG")

    print(f"Overlays generated and saved in {output_folder}")

print("librerias cargadas del chill")



project = "project3-tol0.5 - mergeos y borrados"
csvtogetfrom = pd.read_csv(r"C:\Users\SIMON\Desktop\CÓDIGO FACE TRACKING DE LO DE BOBBYDRAKE\PythonProject\written_ddbbs\bbdd_filled1.csv")
generate_filled_copy(project, csvtogetfrom, "procesado.csv")

#
style_params = {
    'box_color': (255, 255, 255),
    'stroke_width': 3,
    'text_color': (255, 255, 255),
    'font_path': r'C:\Users\simon\Desktop\DOCUMENTAL PERIFERIA CENTRO\Orbitron-VariableFont_wght.ttf',  # Ensure you have the correct font file path
    'font_size': 15,  # Increased font size for 2K video
    'line_spacing': 2,
    'letter_spacing': 2
}


faces_data = r"C:\Users\SIMON\Desktop\CÓDIGO FACE TRACKING DE LO DE BOBBYDRAKE\PythonProject\projects\project3-tol0.5 - mergeos y borrados\data\procesado.csv"
video_file = "Filtered_Facetracking.mp4"


#renderizas en carpeta
generate_overlays(video_file, style_params,faces_data, project)
#
# #añades a davinci y cuadras cada frame donde toca
# #revisas si hay que eliminar en algun nombre algun fotograma por haberse identificado mal
# #borras en la n posicion de la lista de la row con Nombre = x los frames y's
# #renderizas de nuevo
# #posproducción

