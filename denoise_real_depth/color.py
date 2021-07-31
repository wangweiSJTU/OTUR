import cv2
import os.path
import glob
import numpy as np
from PIL import Image
 
def convertPNG(pngfile,outdir):
	# READ THE DEPTH
	im_depth = cv2.imread(pngfile)
	#apply colormap on deoth image(image must be converted to 8-bit per pixel first)
	im_color=cv2.applyColorMap(cv2.convertScaleAbs(im_depth,alpha=0.8),cv2.COLORMAP_JET)
	#convert to mat png
	im=Image.fromarray(im_color)
	#save image
	im.save(outdir)


# convertPNG('9.png','4.png')

path_rgb='22/'
path2='22_c/'
file_list=os.listdir(path_rgb)
for file in file_list:
    pngfile=path_rgb+file
    outdir=path2+file
    convertPNG(pngfile,outdir)


