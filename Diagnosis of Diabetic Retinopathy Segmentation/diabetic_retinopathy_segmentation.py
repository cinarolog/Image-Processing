# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 13:55:19 2022

@author: cinar
"""

#%%

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os




#%%

"""
path'i kendi dosya konumuna göre ayarlayınız!!!
"""
path="C:\\Users\\cinar\\Desktop\\github repo\\image_processing\\Diagnosis of Diabetic Retinopathy Segmentation\\"

df=pd.read_csv(path+"train.csv",sep=",")


df.head()
df.info()

df["diagnosis"].hist()

df["diagnosis"].value_counts()



#%% Görüntüleri okuma işlemi

files=os.listdir(path+"train_images/")
files

len(files)

img_list=[]


for i in files[0:20]:
    
    image=cv2.imread("train_images\\"+i)
    image=cv2.resize(image,(400,400))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img_list.append(image)


len(img_list)


plt.figure(figsize=(6,8))
plt.imshow(img_list[0])
plt.title("Ağrılı nurullah")






#%% Ön İşleme Gaussian Blur


img_list[4].shape
kopya=img_list[4].copy()


kopya=cv2.cvtColor(kopya,cv2.COLOR_RGB2GRAY)
plt.imshow(kopya,cmap="gray")
kopya.shape


blur=cv2.GaussianBlur(kopya, (5,5), 0)
plt.imshow(blur,cmap="gray")


thresh=cv2.threshold(blur,10,255,cv2.THRESH_BINARY)[1]
plt.imshow(thresh,cmap="gray")


#%% Ön İşleme Contours - Görüntüyü kırpma

contour=cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
type(contour)#tuple

contour=contour[0][0]
contour
type(contour)#numpy.ndarray

contour.shape

contour=contour[:,0,:]
contour.shape


#kırpma CROP

contour[:,0].argmax() #335 en sağ
contour[335] #array([332, 218]
contour[:,0].argmin() #111 en sol
contour[111] #array([ 63, 185]

left=tuple(contour[contour[:,0].argmin()])
right=tuple(contour[contour[:,0].argmax()])
top=tuple(contour[contour[:,1].argmin()])
bottom=tuple(contour[contour[:,1].argmax()])

left,right,top,bottom


x1=left[0]
y1=top[1]
x2=right[0]
y2=bottom[1]

x1,y1,x2,y2

original=img_list[4].copy()
plt.imshow(original)


crop_1=original[y1:y2,x1:x2]
plt.imshow(crop_1)
crop_1.shape # (359, 269, 3)

crop_1=cv2.resize(crop_1,(400,400))
plt.imshow(crop_1)
crop_1.shape # (400, 400, 3)


x=int(x2-x1)*4//100
y=int(y2-y1)*5//100
x,y

crop_last=original[ y1+y:y2-y , x1+x:x2-x ]
plt.imshow(crop_last)

crop_last=cv2.resize(crop_last,(400,400))
plt.imshow(crop_last)

#%% CLAHE == Kontrast Limitli Adaptif Histogram Eşitleme

lab=cv2.cvtColor(crop_last, cv2.COLOR_RGB2LAB)
lab.shape

l,a,b=cv2.split(lab)
l,a,b

plt.imshow(l,cmap="gray")
l.shape

flat=l.flatten()
plt.hist(flat,25,[0,256],color="green")

clahe=cv2.createCLAHE(clipLimit=7.0,tileGridSize=((8,8)))
cl=clahe.apply(l)

plt.hist(cl.flatten(),25,[0,256],color="red")
plt.show()

plt.imshow(cl)
plt.title("After from Clahe")

plt.imshow(l)
plt.title("Before from CLAHE")


limg=cv2.merge((cl,a,b))


last=cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
plt.title("After from Clahe  RGB")
plt.imshow(last)


plt.title("Before from CLAHE RGB")
plt.imshow(crop_last)


#%% Median Blur  kanayan kısımları daha ıyı tespit etmemizi saglayacak


median_last=cv2.medianBlur(last, 3)
plt.title("median_last")
plt.imshow(median_last)


background=cv2.medianBlur(last,35)
plt.title("backround")
plt.imshow(background)

masking=cv2.addWeighted(median_last, 1, background, -1, 255)
plt.imshow(masking)
plt.title("masking")

last_img=cv2.bitwise_and(masking,median_last)
plt.imshow(last_img)
plt.title("after from masking/last_img")

plt.imshow(median_last)
plt.title("before from masking/last_img")







