import numpy as np
import scipy.ndimage as nd
import cv2
import os
def fill_img(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    #import numpy as np
    #import scipy.ndimage as nd

    if invalid is None: 
        invalid = data ==0

    dt = cv2.distanceTransform(invalid.astype(np.uint8),cv2.DIST_L2,3)
    # cv2.imshow("re",invalid.astype(np.uint8)*255)
    # cv2.waitKey(0)
    ind = nd.morphology.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

path='depthbin/test/'
save_path='depthbin/zuijinlin_test/'
data_list=os.listdir(path)
for name in data_list:

    im=cv2.imread(path+name,0)

    a=fill_img(im)
    cv2.imwrite(save_path+name,a)
# print(a)
# cv2.imshow('aa',a)
# cv2.waitKey(0)