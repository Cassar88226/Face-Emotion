from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
import sys

class_gaps_y = 50
class_gaps_x = 10
bar_pos_x = 200
bar_width = 300
bar_height = 30
normal_color = (255, 0, 0)
max_color    = (0, 0, 255)
text_color   = (255, 255, 255)
back_color   = (0, 0, 0)



face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

classifier = load_model('testmodel.h5')
class_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

video_file = sys.argv[1]
cap = cv2.VideoCapture(video_file)

while True:
    total_preds = [0] * 7
    ret,frame=cap.read()
    if not ret:
        continue
    labels=[]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)


            preds=classifier.predict(roi)[0]
            total_preds = [sum(x) for x in zip(total_preds, preds)]
    pos_x = 20
    pos_y = 50
    if len(faces) > 0:
        total_preds = [x / len(faces) for x in total_preds]
        cv2.putText(frame, "Emotions", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 255, 0),2)
        for index, label in enumerate(class_labels):
            pos_y += class_gaps_y
            draw_color = normal_color
            pred_max = np.array(total_preds).argmax()
            if label == class_labels[pred_max]:
                draw_color = max_color
            rect_pos_y = pos_y - 25
            cv2.putText(frame, label, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX,1,draw_color,2)
            cv2.rectangle(frame, (bar_pos_x, rect_pos_y), (bar_pos_x + bar_width, rect_pos_y + bar_height), draw_color, 1)
            cv2.rectangle(frame, (bar_pos_x, rect_pos_y), (bar_pos_x + int(bar_width * total_preds[index]), rect_pos_y + bar_height), draw_color, -1)
            cv2.putText(frame, str(round(100 * round(total_preds[index], 4), 2)), (bar_pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX,1,back_color,4)
            cv2.putText(frame, str(round(100 * round(total_preds[index], 4), 2)), (bar_pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX,1,text_color,2)
    else:
        cv2.putText(frame, "Not found face", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 255, 0),2)
    
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
