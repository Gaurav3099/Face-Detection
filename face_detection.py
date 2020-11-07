import cv2

#intialize classifier
faceCascode=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

vc=cv2.VideoCapture(0)

while True:
	#Capture frame
	_, img=vc.read()

	#get gray version of image
	gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#get coordinates of location of face in picture
	faces=faceCascode.detectMultiScale(gray_img,scaleFactor=1.2,minNeighbors=5,minSize=(50,50))

	#Draw a rectangle at location
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+y, y+h),(0,255,0),2)

	#Show image
	cv2.imshow("Identified Face",img)

	#Wait for user to press q
	if cv2.waitKey(1) &0xFF==ord('q'):
		break

#close camera
vc.release()
#close all windows
cv2.destroyAllWindows()