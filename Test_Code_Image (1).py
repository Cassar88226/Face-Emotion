from turtle import pos
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

import sys


face_classifier=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

classifier = load_model('testmodel.h5')
class_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

class_gaps_y = 50
class_gaps_x = 10
bar_pos_x = 200
bar_width = 300
bar_height = 30
normal_color = (0, 0, 255)
max_color    = (255, 0, 0)
text_color   = (255, 255, 255)
back_color   = (0, 0, 0)
total_preds = [0] * 7

img = sys.argv[1]
frame = cv2.imread(img)
face = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces=face_classifier.detectMultiScale(gray,1.1,4)
if len(faces) == 0:
    face = cv2.imread(img)
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
        print(class_labels, preds)
        total_preds = [sum(x) for x in zip(total_preds, preds)]
        label_position=(x,y)
        cv2.rectangle(face,(x,y),(x+w,y+h),normal_color,2)
pos_x = 20
pos_y = 50
total_preds = [x / len(faces) for x in total_preds]
print(len(faces), total_preds)
cv2.putText(face, "Emotions", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 255, 0),2)
for index, label in enumerate(class_labels):
    pos_y += class_gaps_y
    draw_color = normal_color
    pred_max = np.array(total_preds).argmax()
    if label == class_labels[pred_max]:
        draw_color = max_color
    rect_pos_y = pos_y - 25
    cv2.putText(face, label, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX,1,draw_color,2)
    cv2.rectangle(face, (bar_pos_x, rect_pos_y), (bar_pos_x + bar_width, rect_pos_y + bar_height), draw_color, 1)
    cv2.rectangle(face, (bar_pos_x, rect_pos_y), (bar_pos_x + int(bar_width * total_preds[index]), rect_pos_y + bar_height), draw_color, -1)
    cv2.putText(face, str(round(100 * round(total_preds[index], 4), 2)), (bar_pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX,1,back_color,4)
    cv2.putText(face, str(round(100 * round(total_preds[index], 4), 2)), (bar_pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX,1,text_color,2)

imgplot = plt.imshow(face)
plt.show()
