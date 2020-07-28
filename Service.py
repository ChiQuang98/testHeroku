import pickle
from model import *
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
filename = 'data_model/model_SVM.sav'
loaded_model = pickle.load(open(filename, 'rb'))
pretrain_model = load_Pretrain_model()
def load_model_face_detector():
    model = cv2.dnn.readNetFromCaffe("data_model/deploy.prototxt.txt",
                                     "data_model/res10_300x300_ssd_iter_140000.caffemodel")
    return model
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

path_image = "C:/Users/Admin/PycharmProjects/face_model/quang.jpg"
# model_fd = load_model_face_detector()
# classify_model = loaded_model

def face_detector_by_image(imagePath,model_fd,classifier_model,pretrain_model):
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

                listPerson.append(number)

                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, str(number), (startX, startY - 10),
                            cv2.FONT_HERSHEY_PLAIN, 3,
                            (66, 53, 243), 2)
                # cv2.putText(image, str(accuracy), (endX, endY + 10),
                #             cv2.FONT_HERSHEY_PLAIN, 3,
                #             (66, 53, 243), 2)
    cv2.imwrite("savedImage.jpg",image)
    return listPerson
# face_detector_by_image(path_image,model_fd,classify_model,pretrain_model)
