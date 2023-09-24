import os
from flask import Flask, render_template, request
from utils.pipeline import load_image, make_prediction
from utils import *
import matplotlib.pyplot as plt
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'static\uploads'
app.config['OUTPUT_FOLDER'] = r'static\outputs'
app.config['ORGING_FOLDER'] = r'static\origins'

af_pred_to_label = {0:"Ankle",1:"Foot"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files_ap = request.files.getlist('ap_file')
    uploaded_files_lateral = request.files.getlist('lateral_file')
    uploaded_files_oblique = request.files.getlist('oblique_file')

    image_paths = []

    for i, files in enumerate([uploaded_files_ap, uploaded_files_lateral, uploaded_files_oblique]):
        for file in files:
            if file and file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'image_{i}.jpg')
                file.save(file_path)
                image_paths.append(file_path)
    print(image_paths)
    predictions_and_heatmaps = []
    
    for img_path in image_paths:
        view_type = int(img_path.split('_')[-1].split('.')[0])
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f'image_{view_type}.jpg')
        org_path = os.path.join(app.config['ORGING_FOLDER'], f'image_{view_type}.jpg')

        image = load_image(img_path, (256,256), True)
        pred = af_classifier.predict(image)
        label = af_pred_to_label[round(pred[0][0])]
        model, origin = load_model_for_view(label, view_type, img_path)
        prediction, heatmap = make_prediction(model, origin)
        heatmap = cv2.resize(heatmap[0], (700, 1000))
        origin = cv2.resize(origin[0], (700, 1000))

        cv2.imwrite(org_path, origin)
        cv2.imwrite(out_path, heatmap)
        if prediction != "Normal":
            predictions_and_heatmaps.append({
                'prediction': prediction,
                'heatmap': out_path
            })
        else:
            predictions_and_heatmaps.append({
                'img_path': org_path,
                'prediction': prediction,
                'heatmap': org_path
            })            

    return render_template('results.html', results=predictions_and_heatmaps)

def load_model_for_view(label, view_type, img_path):
    if label == "Ankle":
        if view_type == 0:
            origin = load_image(img_path, (140, 200))
            return ankle_ap_view , origin

        elif view_type == 1:
            origin = load_image(img_path, (140, 200))
            return ankle_lateral_view, origin

        elif view_type == 2:
            origin = load_image(img_path, (140, 200))
            return ankle_oblique_view, origin

    elif label == "Foot":
        if view_type == 0:
            origin = load_image(img_path, (140, 200))
            return foot_ap_view, origin
        elif view_type == 1:
            origin = load_image(img_path, (200, 120))
            return foot_lateral_view, origin
        elif view_type == 2:
            origin = load_image(img_path, (140, 200))
            return foot_oblique_view, origin

        

if __name__ == '__main__':
    app.run(debug=True)