from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
# from keras.preprocessing import image
import cv2
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import shutil
import requests



face_classifier=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

classifier = load_model('testmodel.h5')
class_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

img_url = sys.argv[1]
response = requests.get(img_url, stream=True)
img_path = 'test_image.png'
with open(img_path, 'wb') as out_file:
    shutil.copyfileobj(response.raw, out_file)
frame = cv2.imread(img_path)
face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces=face_classifier.detectMultiScale(gray,1.1,4)
if len(faces) == 0:
    plt.title('No Face Was Detected')
    imgplot = plt.imshow(face)
    plt.show()
    exit(0)

for (x,y,w,h) in faces:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray=gray[y:y+h,x:x+w]
    roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

    if np.sum([roi_gray])!=0:
        roi=roi_gray.astype('float')/255.0
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)

        preds=classifier.predict(roi)[0]
        print(preds)
        pred_max = preds.argmax()
        prc = int(100 * round(preds[pred_max], 2))
        label=f'{class_labels[pred_max]} - {prc}%'
        print(label)
        label_position=(x,y)
        cv2.rectangle(face,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(face,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
imgplot = plt.imshow(face)
plt.show()
