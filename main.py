from flask import Flask, flash, request, redirect, url_for, render_template, send_file
import urllib.request
from PIL import Image
import os
import io
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"

app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def to_find_TKS(user_image):
    grayscale_image = cv2.cvtColor(user_image, cv2.COLOR_RGB2GRAY)
    threshold_value, _ = cv2.threshold(
        grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    height, width = grayscale_image.shape
    kernel_size = (max(3, int(width / 10) | 1), max(3, int(height / 10) | 1))
    sigma = (kernel_size[1] * 0.5 - 1) * 0.2
    return threshold_value, kernel_size, sigma


def perform_bokeh(user_image):
    threshold_value, kernel_size, sigma = to_find_TKS(user_image)
    grayscale_image = cv2.cvtColor(user_image, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(
        grayscale_image, threshold_value, 255, cv2.THRESH_BINARY
    )
    inverted_mask = cv2.bitwise_not(binary_mask)
    blurred_background = cv2.GaussianBlur(user_image, kernel_size, sigma)
    user_image[inverted_mask == 255] = blurred_background[inverted_mask == 255]
    return user_image


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        flash("No file path")
        return render_template("uploaded_image.html")

    file = request.files["file"]

    if file.filename == "":
        flash("No image selected for uploading")
        return render_template("index.html", filename = 'filename')

    if file and allowed_file(file.filename):
        image = Image.open(file)
        image_array = np.array(image)

        # Apply the bokeh effect
        image_with_bokeh = perform_bokeh(image_array)

        image_io = io.BytesIO()
        bokeh_image = Image.fromarray(image_with_bokeh)
        bokeh_image.save(image_io, format="PNG")
        image_io.seek(0)
        
        return send_file(image_io, mimetype='image/png')

    else:
        flash("Allowed image types are - png, jpg, jpeg, gif")
        return render_template("index.html")


@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename="uploads/" + filename), code=301)


if __name__ == "__main__":
    app.run(debug = True)
