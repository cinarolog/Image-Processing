# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 23:45:58 2021

@author: Cinar
"""
#%%

""" # import libraries  """
import cv2 
import numpy as np
import matplotlib.pyplot as plt

# resmi siyah beyaz olarak içe aktaralım
image = cv2.imread("test (1).jpg")
plt.figure(), plt.imshow(image), plt.axis("off"), plt.title("Image Original")

#%% Gray scale Coversion


"""
Burada resmimizi renkli formattan gri formata dönüştürüyoruz
ve görselleştiriyoruz.

"""
# Gray scale Coversion
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(image_gray), plt.axis("off"), plt.title("Gray")
# cv2.imshow('Yaprak',image_gray)
# cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows()

#%%

""" # Gaussian Blur """

"""
Burada resmimize bluring işlemi yapıyoruz yani bulanıklaştırıyoruz
ve görselleştiriyoruz.

"""

image_gb = cv2.GaussianBlur(image_gray, ksize = (3,3), sigmaX = 7)
plt.figure(), plt.imshow(image_gb), plt.axis("off"), plt.title("Gaussian Blur")


#%%

""" # Thresholding """

"""
Burada resmimize eşikleme yapıyoruz renk skalasın 100 ün altında olan piksellerimizi beyaz,
geriye kalanlarını ise siyah yapıyoruz.
Özet geçersek verilen resmi ikili binary görüntüye çevirmek .Siyah-Beyaz 
"""

_, thresh_img = cv2.threshold(image_gb, thresh = 100, maxval = 255, type = cv2.THRESH_BINARY)



# 

plt.figure()
thresh_img = cv2.bitwise_not(thresh_img, mask = None)
# thresh_img = cv2.resize(thresh_img, (227,227), interpolation = cv2.INTER_AREA)
plt.title(" Otsu Thresholding")
plt.imshow(thresh_img, cmap = "gray")

# print(thresh_img.shape)

# %%
""" #Generating blank black image of same dimension """

"""

Burada blank oluştırmak için bir tane fonksiyon tanımlıyoruz
bu fonksiyon rengini istegimize göre ayarlayabilecegimiz  blank olusturacak.
#Generaing blank black image of same dimension

"""


def create_blank(width, height, rgb_color=(0, 0, 0)):
    
    """Create new image(numpy array) filled with certain color in RGB"""
    
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    
    # Fill image with color
    image[:] = color

    return image


# Create new blank 300x300 red image

width1, height1 = image.shape[0], image.shape[1]

black = (0, 0, 0)
thresh_img2 = create_blank(width1, height1, rgb_color=black)
plt.figure(), plt.imshow(thresh_img2)


# %%
""" # contours """

"""
Burada aynı renk ve yoğunluğa sahip olan kesintisiz noktaları sınır boyunca
birleştiren bir eğri oluşturuyoruz. Biz burada dış(external) eğrileri,çizgileri kullandık.

"""

contours, hierarch = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

external_contour = np.zeros(thresh_img.shape)
internal_contour = np.zeros(thresh_img.shape)

for i in range(len(contours)):
    
    # external
    if hierarch[0][i][3] == -1:
        cv2.drawContours(external_contour,contours, i, 255, -1)
    else: # internal
        cv2.drawContours(internal_contour,contours, i, 255, -1)

plt.figure(), plt.imshow(external_contour, cmap = "gray"),plt.axis("off")
#plt.figure(), plt.imshow(internal_contour, cmap = "gray"),plt.axis("off")


"""
#plt.figure(), plt.imshow(internal_contour, cmap = "gray"),plt.axis("off")
Eğer içten kontür çizdirmek istiyorsanız bunu kullanın

""" 

 
# %% Bitwise Masking 

""" Bitwise Masking  """

"""
Bir maske hangi bitleri saklamak istediğinizi ve hangi bitleri temizlemek istediğinizi tanımlar.
Maskeleme, bir değere maske uygulama eylemidir. Bu, aşağıdakileri yaparak gerçekleştirilir:

*Değerdeki bitlerin bir alt kümesini çıkarmak için Bitwise ANDing
*Değerdeki bitlerin bir alt kümesini ayarlamak için Bitwise ORing
*Değerdeki bitlerin bir alt kümesini değiştirmek için Bitsel XORing

Sayıların ikilik (binary), onluk (decimal) ve on altılık (hexadecimal) 
tabanda ifade edilişleri ve maskeleme yardımıyla veri işleme yöntemleri.

"""
#
# thresh_img2 = thresh_img2.reshape(thresh_img2.shape[0],thresh_img2.shape[1]*thresh_img2.shape[2])
# thresh_img2 = cv2.resize(thresh_img2, (64,64), interpolation = cv2.INTER_AREA)
print(external_contour.shape)
print(image.shape)
# external_contour = external_contour.reshape(256,256,1)
print(external_contour.shape)
mask = np.zeros((256,256), dtype=np.uint8)
mask = cv2.circle(mask, (256,256), 225, (255,255,255), -1) 
print(mask.shape)
# Mask input image with binary mask
result = cv2.bitwise_and(thresh_img,thresh_img,image,mask=None)
cv2.imshow('Yaprak',result)
cv2.waitKey(1)                        
# Color background white
#result[mask==0] = 255 # Optional
plt.show()
plt.imshow(result)
plt.title("Final Segmented Image")
plt.show()

print()
print("************************************")
print("Segmentation has been successfully applied.")
print("************************************")
print()

#result

cv2.waitKey()
cv2.destroyAllWindows()


#%%%

