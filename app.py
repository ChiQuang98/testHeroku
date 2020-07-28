from flask import Flask, request, render_template, jsonify

from Service import *
import pandas as pd

import threading
# implementFaceDetect(model,classifier_model,vgg_face)
from flask import Flask, request, render_template
model_fd = load_model_face_detector()
classify_model = loaded_model
app = Flask(__name__)
pretrain_model = load_Pretrain_model()
@app.route('/upload', methods=['POST'])
def uploadFile():
    global person_rep
    # person_rep  = load_person_rep(pathPerson_Rep)
    file = request.files.get('file')
    if file and file.filename != '':
        file.save("DataClient/"+file.filename)
        listJSON = face_detector_by_image("DataClient/"+file.filename,model_fd,classify_model,pretrain_model)
    # print((jsonify(listJSON)))
    return pd.Series(listJSON).to_json(orient='values')

@app.route('/')
def homePage():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)