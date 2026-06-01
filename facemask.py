import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import cv2
import datetime

mymodel = load_model('mymodel.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg', face_img)

        test_image = load_img('temp.jpg', target_size=(150, 150))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        pred = mymodel.predict(test_image)[0][0]
        print("Prediction value:", pred)
        if pred >0.5:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(img, 'NO MASK', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img, 'MASK', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    datet = str(datetime.datetime.now())
    cv2.putText(img, datet, (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Mask Detection', img)

    # ↓↓↓ correct position (inside while loop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ↓↓↓ loop se bahir
cap.release()
cv2.destroyAllWindows()