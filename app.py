import os
import sys
print(sys.executable)

from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import Sequential, model_from_json
import tensorflow as tf
import librosa
import numpy as np
import math
from tcn import TCN, tcn_full_summary

from werkzeug.utils import secure_filename
import numpy as np

ALLOWED_EXTENSIONS = set(['wav'])
UPLOAD_FOLDER = 'uploads'

def load_model():
    loaded_json = open('model.json', 'r').read()
    model = model_from_json(loaded_json, custom_objects={'TCN': TCN})
    model.load_weights('weights.h5')
    return model

model = load_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def predict(file):
    waveform, sample_rate = librosa.load(file, sr=22050, duration=4.0)

    # make sure that the audio is sample_rate * 4 seconds long
    if len(waveform) != (sample_rate * 4):
        repeat_num = math.ceil((sample_rate*4) / len(waveform))
        waveform = np.repeat(waveform, repeat_num)
        waveform = waveform[:(sample_rate*4)]

    matrix = librosa.stft(waveform, center=False, n_fft=1024, hop_length=256)

    matrix = np.stack([np.real(matrix), np.imag(matrix)], -1)
    matrix = np.swapaxes(matrix, 0, 1)
    matrix = np.expand_dims(matrix, axis=0)
    output = model.predict(matrix)
    output = np.squeeze(output)

    output = {
        "air_conditioner": output[0],
        "car_horn": output[1],
        "children_playing": output[2],
        "dog_bark": output[3],
        "drilling": output[4],
        "engine_idling": output[5],
        "gun_shot": output[6],
        "jackhammer": output[7],
        "siren": output[8],
        "street_music": output[9]
    }
    return output

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['input-b1']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
            return render_template("home.html", label=output, audiosource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
