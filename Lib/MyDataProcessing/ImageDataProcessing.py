
# coding: utf-8

# In[11]:


import cv2
import os
import numpy
import matplotlib.pyplot as plt

def GetImageFromDirec(DirecPath):
    fileList = os.listdir(DirecPath)
    fileImgs = []
    for file in fileList:
        img = cv2.imread(DirecPath + '/' +file, cv2.IMREAD_COLOR)
        fileImgs.append(img)
    return fileImgs

def GetImageFromFile(FilePath):
    img = cv2.imread(FilePath, cv2.IMREAD_COLOR)
    return img

def ImageResizing(Images,size):
    resizeImgs = []
    for img in Images:
        result = cv2.resize(img,size)
        resizeImgs.append(result)
    return resizeImgs
        
def ListToNumpy(data):
    return numpy.array(data)

def ImageShow_plt(image,title=None,option=cv2.COLOR_BGR2RGB,size=(25,5)):
    if(title != None):
        print(title)	
    plt.rcParams["figure.figsize"] = size
    plt.imshow(cv2.cvtColor(image, option))
    plt.show()
   
def RGBToGray(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray

def RGBToGray_imgs(imgs):
    GrayImgs = []
    for img in imgs:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        GrayImgs.append(gray)
    return GrayImgs
	
# imageList = GetImageFromDirec('./[Data] Crack Data/(3) CrackForest-dataset-master/CrackForest-dataset-master/image')
# imageList = ListToNumpy(imageList)
# imageList.shape

# re_images = ImageResizing(imageList,256,256)
# re_images = ListToNumpy(re_images)
# re_images.shape

# ImageShow_plt(re_images[1])
