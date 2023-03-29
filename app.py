import os
import cv2
import pytesseract
import numpy as np
import img2pdf
from typing import Tuple
from flask import Flask, render_template, request, send_from_directory, send_file, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from flask_socketio import SocketIO
from flask import Flask, render_template, request
from flask_cors import CORS




pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# make this code more efficient
app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


# Set the directory to store the extracted frames
frame_dir = 'static/frames/'

# Delete previous frames if the directory exists
if os.path.exists(frame_dir):
    for file_name in os.listdir(frame_dir):
        file_path = os.path.join(frame_dir, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")

# Create the directory to store the extracted frames
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)


# Constants
FPS_LIMIT = 30
MAX_FRAMES = FPS_LIMIT * 10  # Limit to 10 seconds of frames (300 frames for 30 fps video)
MIN_LAP_VAR = 100
SSIM_THRESHOLD = 0.9
RESIZE_WIDTH = 800

# Function to extract frames from a video file
def extract_frames(video_file_path: str):
    global frame_dir


    

    # Read the uploaded video file
    vidcap = cv2.VideoCapture(video_file_path)

    # Set the frame rate and counter
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    if fps > FPS_LIMIT:
        print(f"WARNING: Frame rate of {fps} fps exceeds limit of {FPS_LIMIT} fps. Processing may be slower than expected.")
    count = 0

    # Extract frames and save as images
    success, image = vidcap.read()
    prev_image = None
    while success and count < MAX_FRAMES:
        # Check if current frame is the same as previous frame
        if prev_image is not None and np.array_equal(image, prev_image):
            success, image = vidcap.read()
            continue

        # Resize image to speed up comparison
        image_resized = cv2.resize(image, (RESIZE_WIDTH, RESIZE_WIDTH))

        # Check for duplicate pages
        if count > 0:
            ssim_val = ssim(image_resized, prev_image_resized, multichannel=True, win_size=3)
            if ssim_val > SSIM_THRESHOLD:
                success, image = vidcap.read()
                continue

        # Check for blurry or blocked pages
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi_size = min(h, w) // 2
        roi_y = (h - roi_size) // 2
        roi_x = (w - roi_size) // 2
        roi = gray[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        lap_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        if lap_var < MIN_LAP_VAR:
            success, image = vidcap.read()
            continue

        # Rotate the image if the width is greater than the height
        if w > h:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


        # Save the frame as an image
        frame_path = os.path.join(frame_dir, f'frame_{count}.jpg')
        cv2.imwrite(frame_path, image.astype(np.uint8))

        # Move to the next frame
        count += 1
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, count * fps)
        success, image = vidcap.read()
        if image is not None:
            prev_image_resized = image_resized.copy()
            prev_image = image.copy()  # Save previous image for comparison

    # Use OCR to extract text from the frames
    ocr_text = ''
    for i in range(count):
        # Load the image
        frame_path = os.path.join(frame_dir, f'frame_{i}.jpg')
        image = cv2.imread(frame_path)

        

        # Extract the document from the image
        document, text = extract_document(image)
        ocr_text += text

        # Save the document as an image
        doc_path = os.path.join(frame_dir, f'document_{i}.jpg')
        cv2.imwrite(doc_path, document.astype(np.uint8))

    # Convert the extracted text to PDF
    pdf_path = os.path.join(frame_dir, 'output.pdf')
    with open(pdf_path, 'w+b') as f:
        f.write(img2pdf.convert([open(os.path.join(frame_dir, f'document_{i}.jpg'), 'rb') for i in range(count)]))



    # Return the extracted text and the PDF file for download
    return send_from_directory(directory=os.path.abspath(frame_dir), path='output.pdf', as_attachment=True), ocr_text










def extract_document(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Tesseract OCR to extract text from the image
    ocr_text = pytesseract.image_to_string(gray)

    return image, ocr_text




@app.route('/')
def index_html():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['pc-upload']
    file_path = 'static/uploads/' + file.filename
    file.save(file_path)

    # Pass the file path to the 'success.html' template
    return redirect(url_for('animation', video_file_path=file_path))


@app.route('/animation')
def animation():
    return render_template('animation.html')

@app.route('/success', methods=['GET'])
def success():
    # Get the file path of the uploaded file from the request args
    video_file_path = request.args.get('video_file_path')

    # Extract video frames
    pdf_file, ocr_text = extract_frames(video_file_path)

    # Delete uploaded video file
    os.remove(video_file_path)

    # Pass the extracted frames to the 'success.html' template
    return render_template('success.html', pdf_file=pdf_file, ocr_text=ocr_text)



@app.route('/download_pdf')
def download_pdf():
    pdf_path = os.path.join(app.root_path, 'static/frames/output.pdf')
    return send_file(pdf_path, as_attachment=True)




if __name__ == "__main__":
    app.run(debug=True)             