from PIL import Image
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
model = load_model('face_recog.h5')
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img 
from tensorflow.keras.applications.vgg16 import preprocess_input

# Loading the cascades
face_identifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    faces = face_identifier.detectMultiScale(img, 1.3, 5)
    
    if faces == ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,128,0),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Doing some Face Recognition with the webcam
cap = cv2.VideoCapture(0)
while True:
        preds=[]
        _, frame = cap.read()
        face=face_extractor(frame) 
        if face != []:
            face = cv2.resize(face ,(224,224))
            inp = np.asfarray(face).astype('float32')
            print(inp.shape,inp.dtype)
            inp=inp/255.0
            inp= np.expand_dims(inp,axis=0)
            preds = model.predict(inp)[0]
            clases = np.argmax(preds)
            print(clases)
            if clases==1:
                name = "Keetu"
                cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (128,0,0), 2)
                

            elif clases==0:
                name ="Vishal"
                cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (128,0,0), 2)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            

            elif clases==2:
                name="Mom"
                cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (128,0,0), 2)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                    cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            

        else:
                cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
    
cap.release()
cv2.destroyAllWindows()