from flask import Flask, request, render_template, jsonify

# from Service import *
from model import *
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import threading
# implementFaceDetect(model,classifier_model,vgg_face)
from flask import Flask, request, render_template
model_fd = load_model_face_detector()
classify_model = load_model()
pretrain_model = load_Pretrain_model()
labels = load_Labels()
le =  LabelEncoder()
labelsEnc = le.fit_transform(labels)
app = Flask(__name__)
@app.route('/attendance', methods=['POST'])
def uploadFile():
    global person_rep
    # person_rep  = load_person_rep(pathPerson_Rep)
    file = request.files.get('file')
    if file and file.filename != '':
        file.save("DataClient/"+file.filename)
        listJSON = face_detector_by_image("DataClient/"+file.filename,model_fd,classify_model,pretrain_model,labelsEnc)
    # print((jsonify(listJSON)))
    # return pd.Series(listJSON).to_json(orient='values')
    return jsonify(listJSON)

@app.route('/')
def homePage():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='192.168.1.103', port=5000)