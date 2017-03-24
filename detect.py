import cv2
import os
import numpy as np
from PIL import Image
#detects faces
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
recognizer = cv2.createLBPHFaceRecognizer()
# Get the images from the database
def getimages(path):
	image_paths = [os.path.join(path,f) for f in os.listdir(path) if not f.endswith('.sad')]
	images = []
	labels = []
	for image_path in image_paths:
		image_pil = Image.open(image_path).convert('L')
		image = np.array(image_pil,'uint8')
		nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject",""))
		faces = faceCascade.detectMultiScale(image)
		for (x,y,w,h) in faces:
			images.append(image[y:y+h,x:x+w])
			labels.append(nbr)
			cv2.imshow("Adding faces to training set..",image[y:y+h,x:x+w])
			cv2.waitKey(50)
	return images, labels

path = 'faces'
images, labels = getimages(path)
cv2.destroyAllWindows()
#Train the dataset
recognizer.train(images,np.array(labels))
#Testing 
image_paths = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.sad')]
for image_path in image_paths:
	predict_PIL = Image.open(image_path).convert('L')
	predict_image = np.array(predict_PIL,'uint8')
	faces = faceCascade.detectMultiScale(predict_image)
	for(x,y,w,h) in faces:
		nbr_predict, confidence = recognizer.predict(predict_image[y:y+h,x:x+w])
		nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject",""))
		if nbr_actual == nbr_predict:
			print "Correctly recognized with Confidence {}".format(confidence)
		else:
			print "Wrongly recognized with Confidence {}".format(confidence)
		cv2.imshow("Recognizing face",predict_image[y:y+h,x:x+w])
		cv2.waitKey(1000)
