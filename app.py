import cv2         #opencv Library 
import numpy 



#	key = cv2.waitKey(1)      #this line is for termination 
cap = cv2.VideoCapture(0)    # this line is use to access the camera  and 0 means inbuilt camera use 1 for external camera 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 


while True:
	ret , frame = cap.read()       # cap return the  two values ret and frame ret contains tha matrix and frame contain tha frames
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #here we convert the color img in gray image for better op 
														# in colour image there are 3 channels and in grey thre are only 1 
	
	faces_in_image = face_cascade.detectMultiScale(gray, 1.3, 5)   # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
    														#minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it

	
	for (x,y,w,h) in faces_in_image:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
		


	cv2.imshow('Captured image',frame)     #show the gui and camera op
	key =cv2.waitKey(1)
	if key==ord('p'):         # if we press p it quit automatically 
		cap.release()
		cv2.destroyAllWindows()
