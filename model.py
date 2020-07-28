# Thêm thư viện
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imutils import paths
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
from keras_vggface.vggface import VGGFace
import pickle
# from sklearn.svm import SVC
from sklearn import svm
import tensorflow.keras.backend as K


def load_Pretrain_model():
    # Load model VGG 16 của ImageNet dataset, include_top=False để bỏ phần Fully connected layer ở cuối.
    model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')  # pooling: None, avg or max
    # model.summary()
    return model
def getFeatureImage(faceImage,model):
    crop_img = img_to_array(faceImage)
    crop_img = np.expand_dims(crop_img, axis=0)
    crop_img = imagenet_utils.preprocess_input(crop_img)
    features = model.predict(crop_img)

    return features

def loadFeature():
    features = np.load('data_model/features_VN-celeb_SVM.npy')
def loadLebels():
    labels = np.load('data_model/labels_SVM.npy')
def splitTrainingSet(features,labels):
    # Chia traing set, test set tỉ lệ 80-30
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=20)
def buildModel(X_train,y_train,X_test,y_test):
    # Use SVM
    model = svm.SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    # Grid search để tìm các parameter tốt nhất cho model. C = 1/lamda, hệ số trong regularisation. Solver là kiểu optimize
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    params = {'C': [0.1, 1.0, 10.0, 100.0]}
    # model = GridSearchCV(LogisticRegression(solver='lbfgs', multi_class='multinomial'), params)
    model = GridSearchCV(LogisticRegression(), params)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print('Best parameter for the model {}'.format(model.best_params_))
    # Đánh giá model
    preds = model.predict(X_test)
    filename = 'data_model/model_SVM.sav'
    pickle.dump(model, open(filename, 'wb'))
    print("Do chinh xac: " + str(score))
    # score = model.evaluate
    print(classification_report(y_test, preds))
# Make Predictions
def predict(img_encode,classifier_model):
  embed = K.eval(img_encode)
  person = classifier_model.predict(embed)#day la so 136
  # le = LabelEncoder()
  # accurary = np.max(person)
  # print(le.fit_transform(person))
  return person

