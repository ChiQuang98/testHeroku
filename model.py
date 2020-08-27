import tensorflow as tf
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras

import cv2
from keras.applications import VGG16
from sklearn.preprocessing import LabelEncoder
from keras_vggface.vggface import VGGFace

def load_Pretrain_model():
    # Load model VGG 16 của ImageNet dataset, include_top=False để bỏ phần Fully connected layer ở cuối.
    model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')  # pooling: None, avg or max
    # model.summary()
    return model
def load_model():
    # tf.keras.models.load_model('data_model/face_recognition_restnet50_train_val_test.h5')
    model =  tf.keras.models.load_model('data_model/face_recognition_restnet50_train_val_test.h5')
    return model
def load_model_face_detector():
    model = cv2.dnn.readNetFromCaffe("data_model/deploy.prototxt.txt",
                                     "data_model/res10_300x300_ssd_iter_140000.caffemodel")
    return model
def getFeatureImage(faceImage,model):
    crop_img = img_to_array(faceImage)
    crop_img = np.expand_dims(crop_img, axis=0)
    crop_img = imagenet_utils.preprocess_input(crop_img)
    features = model.predict(crop_img)
    return features
def load_Labels():
    label_person = []
    # open file and read the content in a list
    with open('data_model/label.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPerson = line[:-1]
            # add item to the list
            label_person.append(currentPerson)
    return label_person
# Make Predictions
def cut_faces(image,faces_coord):
    (x,y,w,h) =  faces_coord
    w = w-x
    h = h-y
    w_rm = int(0.2*w/2)
    faces = (image[y:y+h,x+w_rm:x+w-w_rm])
    return faces
def resize(image,size = (224,224)):
    if image.shape < size:
        image_norm = cv2.resize(image, size, 3)
    else:
        image_norm = cv2.resize(image, size, 3)
    return image_norm
def face_detector_by_image(imagePath,model_fd,classifier_model,pretrain_model,labelEnc):
    listPerson = []
    image = cv2.imread(imagePath)
    # print(image)
    # faces_coord = detector.detect(frame) #faces_coord is numpy array,day la toa do guong mat ma minh lay duoc
    (h, w) = image.shape[:2]
    # blobImage convert RGB (104.0, 177.0, 123.0)
    # blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104.0, 177.0, 123.0))
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0, (224, 224), (104.0, 177.0, 123.0))
    model_fd.setInput(blob)
    detections = model_fd.forward()
    # print(len(faces_coord))#face_coord la mot mang luu cac toa do khuong mat tren frame
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # print(j)
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # extract the confidence and prediction
        confidence = detections[0, 0, i, 2]
        if (confidence > 0.165):
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            face = cut_faces(image, box.astype("int"))  # tu toa do ta se cat duoc cac khuon mat

            if len(face):
                face = resize(face)
                cv2.imwrite("maianhCrop.jpg", face)
                global name, accuracy
                img_encode = getFeatureImage(np.reshape(face, (224, 224, 3)), pretrain_model)
                number = predict(img_encode, classifier_model)
                le = LabelEncoder()
                msv = le.inverse_transform(number)
                listPerson.append(str(number))

                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, str(msv), (startX, startY - 10),
                            cv2.FONT_HERSHEY_PLAIN, 3,
                            (66, 53, 243), 2)
                # cv2.putText(image, str(accuracy), (endX, endY + 10),
                #             cv2.FONT_HERSHEY_PLAIN, 3,
                #             (66, 53, 243), 2)
    cv2.imwrite("savedImage.jpg",image)
    return listPerson
def predict(img_encode,classifier_model):
  embed = K.eval(img_encode)
  person = classifier_model.predict(embed)#day la so 136
  y = np.argmax(person, axis=1)
  # le = LabelEncoder
  # le.inverse_transform(y)
  # le = LabelEncoder()
  # accurary = np.max(person)
  # print(le.fit_transform(person))
  return y

