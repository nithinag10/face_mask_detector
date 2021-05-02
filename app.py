from flask import Flask, request, jsonify, render_template, Response
import cv2
import numpy as np
import jsonpickle
from PIL import Image

from util import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    r = request
    arr = np.frombuffer(r.data, np.uint8)
    img= cv2.imdecode(arr, cv2.IMREAD_COLOR)
    data = Image.fromarray(img)
    tensor = transform_image(data)
    prediction = get_prediction(tensor)
    data = {'prediction': prediction}
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)