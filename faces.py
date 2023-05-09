import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture('data/obama2.mp4')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("data/model/face-trainner.yml")

labels = {"person_name": 1}
with open("data/model/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

while (1):

    # capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        img_item = "my-image.png"
        #cv2.imwrite(img_item, roi_color)

        id_, conf = recognizer.predict(roi_gray)
        if(conf>=45):
            print(labels[id_])
            print('-')
            font = cv2.FONT_HERSHEY_SIMPLEX
            name  = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y), font,1,color,stroke,cv2.LINE_AA)

        color = (255, 0, 0)  # BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h

        cv2.rectangle(frame, (x, y),(end_cord_x,end_cord_y),color,stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# when everything done, release the capture
cap.release()
cv2.destroyAllWindows()
