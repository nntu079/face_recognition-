import os
from PIL import Image
import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "data/train")

y_labels = []
x_train = []
current_id = 0
label_ids = {}


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).lower()

            # y_labels.append(label)
            # x_train.append(path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]         
            

            pil_image = Image.open(path).convert("L")
            size = (550, 550)

            final_image = pil_image.resize(size,Image.ANTIALIAS)

            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.5, minNeighbors=5)
            
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("data/model/face-trainner.yml")

with open("data/model/face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)