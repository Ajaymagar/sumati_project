from tensorflow.keras.models import load_model
import cv2
import numpy as np
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
maskNet = load_model('mask_detector.model')




def detectAndPredict(frame):
    locs = []
    faces = []
    preds = []
    faces_in_image = face_cascade.detectMultiScale(frame , 1.3 ,5)
    for (x ,y,w,h) in faces_in_image:
        locs.append((x,w,y,h))
        faces.append((x+w,y+h))

    faces = np.array(faces ,dtype="float32")
    preds= maskNet.predict(faces, batch_size=32)
    return (locs , preds)

while True:
    success, image = video.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (locs , preds) = detectAndPredict(gray)
	
    for (box , pred) in zip(locs , preds):
        (x,y,w,h) = box
        (mask , withoutmask) = pred

        label ="Mask" if mask > withoutmask else "No mask"
        if label == "Mask":
            color  = (0 ,255 ,0)
        else:
	        color = (0,0,255)

        label  = "{}:{:.2f}%".format(label,max(mask ,withoutmask)*100)
        
        cv2.putText(image , label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image,(x,y),(w,h),color,2)

        cv2.imshow('frame',image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            break

