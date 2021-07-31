import cv2
import numpy as np
import os

path = ('depth/')
target = ('depth_noise/')
file_list=os.listdir(path)
for file in file_list:
	image=cv2.imread(path+'/'+file)
	noise_sigma = 50
	noise = np.random.normal(size=image.shape)*noise_sigma
	noise = cv2.GaussianBlur(noise,(5,5),0)
	noise_img = image + noise
	cv2.imwrite(target+'/'+file,noise_img)
