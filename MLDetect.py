import os
import numpy as np 
from PIL import Image
from hog import hog
from skimage import exposure
from sklearn import svm

def getimages(path):
	image_paths = [os.path.join(path,f) for f in os.listdir(path) if not f.endswith('.sad')];
	images = [];
	names = [];
	data = [];
	for image_path in image_paths:
		grey = Image.open(image_path).convert('L');
		image = np.array(grey,'uint8')
		subject = int(os.path.split(image_path)[1].split(".")[0].replace("subject",""))
		hist, hog_img = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)
		hog_image = Image.fromarray(hog_img,'RGB')
		images.append(hog_image)
		names.append(subject)
		data.append(hist)
	return images,names,data

def getpics(path):
	image_paths = [os.path.join(path,f) for f in os.listdir(path) if not f.endswith('10.jpg')];
	images = [];
	names = [];
	data = [];
	for image_path in image_paths:
		grey = Image.open(image_path).convert('L');
		image = np.array(grey,'uint8')
		subject = os.path.split(image_path)[1].split(".")[0].split("0")[0];
		hist, hog_img = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True);
		hog_image = Image.fromarray(hog_img,'RGB');
		images.append(hog_image);
		names.append(subject);
		data.append(hist);
	return images,names,data;

def testimages(path):
	image_paths = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.sad')];
	for image_path in image_paths:
		grey = Image.open(image_path).convert('L');
		image = np.array(grey,'uint8');
		actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject",""));
		hist, hog_img = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True);
		hog_image = Image.fromarray(hog_img,'RGB');
		prediction = model.predict(hist)[0];
		if actual == prediction :
			print("Correct prediction");
		else:
			print("Wrong prediction");

def testpics(path):
	image_paths = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('10.jpg')];
	for image_path in image_paths:
		grey = Image.open(image_path).convert('L');
		image = np.array(grey,'uint8');
		actual = os.path.split(image_path)[1].split(".")[0].split("1")[0];
		hist, hog_img = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True);
		hog_image = Image.fromarray(hog_img,'RGB');
		prediction = model.predict(hist)[0];
		if actual == prediction :
			print('Actual:{} Predicted:{} Correct prediction'.format(actual,prediction));
		else:
			print('Actual:{} Predicted:{} Wrong prediction'.format(actual,prediction));

path = 'faces';
images, names, data = getimages(path);
model = svm.SVC(C=100.0,random_state=42);
model.fit(data,names);
#testpics(path);

