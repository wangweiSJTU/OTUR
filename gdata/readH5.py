import h5py
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import randint, seed

path="tr_depth32/"
save_path="masked/"
data_list=os.listdir(path)
hole_sigma=0.03

def generate_hole(height=256, width=256, channels=1):
    """Generates a random irregular mask with lines, circles and elipses"""

    img = np.zeros((height, width,channels), np.uint8)

    # Set size scale
    size = int((width + height) * hole_sigma)
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")
        
    # Draw random lines
    for _ in range(randint(1, 10)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        thickness = randint(3, size)
        cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
            
        # Draw random circles
    for _ in range(randint(1, 10)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(3, size)
        cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
            
    # Draw random ellipses
    for _ in range(randint(1, 10)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, size)
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)

    img=img.transpose(2,1,0)
        
    return 1-img


# for name in data_list:
# 	dataset = h5py.File(path+name, 'r')
# 	dataset_new = h5py.File(save_path+name, 'w')
# 	data = np.array(dataset["label"][:])
# 	mask=np.zeros(data.shape)

# 	for i in range(data.shape[0]):
# 		mask[i]= generate_hole()

# 	data_new=data*mask
# 	dataset_new['label']=data
# 	dataset_new['data']=data_new
# 	dataset_new['mask']=mask
# 	dataset.close()
# 	dataset_new.close()

dataset1 = h5py.File('tr_depth32/d32_train_00000000.h5', 'r')
dataset2 = h5py.File('tr_depth32/d32_train_00000001.h5', 'r')
dataset_new = h5py.File('masked/d32_train_00000001.h5', 'w')

data = np.array(dataset1["label"][:])
data2 = np.array(dataset2["label"][:])
print(data.shape,data2.shape)
# data2 = data2[:data.shape[0],:,:,:]
if data.shape[0]<=data2.shape[0]:
    data2 = data2[:data.shape[0],:,:,:]
    per = np.random.permutation(data.shape[0])
    data = data[per, :, :, :]
    data2 = data2[per, :, :, :]
else:
    data = data[:data2.shape[0],:,:,:]
    per = np.random.permutation(data2.shape[0])
    data = data[per, :, :, :]
    data2 = data2[per, :, :, :]
# data = data[:data2.shape[0],:,:,:]
print(data.shape)
print(data2.shape)
# per = np.random.permutation(data.shape[0])
# data = data[per, :, :, :]
# data2 = data2[per, :, :, :]
dataset_new['label']=data2
dataset_new['data']=data
dataset1.close()
dataset2.close()
dataset_new.close()