import os
import sys
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import librosa
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

ALLOWED_EXTENSIONS = set(['wav'])
UPLOAD_FOLDER = 'uploads'

def load_model_and_label_encoder():
    model = load_model('audio_classification.hdf5')
    labelencoder_classes = np.load('labelencoder_classes.npy', allow_pickle=True)
    label_encoder = {index: label for index, label in enumerate(labelencoder_classes)}
    return model, label_encoder

model, label_encoder = load_model_and_label_encoder()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def predict(file_path, top_n=3):
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    predicted_probs = model.predict(mfccs_scaled_features)[0]
    top_n_probs_idx = np.argsort(predicted_probs)[::-1][:top_n]
    top_n_probs = predicted_probs[top_n_probs_idx]
    top_n_labels = [label_encoder[top_n_probs_idx[i]] for i in range(top_n)]
    return top_n_labels, top_n_probs

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('home.html', labels='', probs='', imagesource='file://null')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['input-b1']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            labels, probs = predict(file_path, top_n=1)  # Change `top_n` to the desired number of top classes
            return render_template("home.html", labels=labels, probs=probs, audiosource=file_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
