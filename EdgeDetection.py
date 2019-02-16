# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 18:01:59 2018

@author: Himanshu Garg
UBName : hgarg
UB Person : 50292195
"""
import cv2
import numpy as np
import math

def convertToList(nparray):
    nlist = []
    for index,i in enumerate(nparray):
        temp = list(i)
        temp = [x*1.0 for x in list(i)]
        nlist.append(temp)
    return nlist

img = cv2.imread("task1.png",0)
sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]      #flipFilter:sobel_vertical
sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]      #flipFilter:sobel_horizontal

img2list = convertToList(img)
imgxedge = []
imgyedge = []
imgxedgeabs = []
imgyedgeabs = []
mag = []
ori = []
maxMag = 0.0 
maxImgX = 0.0
minImgX = 0.0
maxImgY = 0.0
minImgY = 0.0
maxOri = 0.0
count = 0

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for indxr,row in enumerate(img2list):
    if indxr == 0 or indxr == len(img2list) - 1:
        continue
    tempx = []
    tempy = []
    tempxabs = []
    tempyabs = []
    tempmag = []
    tempori= []
    for indc,col in enumerate(row):
        if indc == 0 or indc == len(row) - 1:
            continue
        
        r1 = img2list[indxr - 1]
        l1 = r1[indc-1:indc+2]
        l2 = row[indc-1:indc+2]
        r3 = img2list[indxr + 1]
        l3 = r3[indc-1:indc+2]
        
        valx = sum([l1[k]*sobel_x[0][k] for k in range(3)]) + sum([l2[m]*sobel_x[1][m] for m in range(3)]) + sum([l3[n]*sobel_x[2][n] for n in range(3)])
        valy = sum([l1[p]*sobel_y[0][p] for p in range(3)]) + sum([l3[q]*sobel_y[2][q] for q in range(3)])
        
        mg = math.sqrt(valx ** 2 + valy ** 2)
        val_ori = (math.atan2(valy,valx + 1e-3)*180)/math.pi
        
        if count == 0:
            maxMag=mg
            maxOri=val_ori
            maxImgX=valx
            maxImgY=valy
            minImgX=valx
            minImgY=valy
            count+=1
        
        if mg > maxMag:
            maxMag = mg
        if val_ori > maxOri:
            maxOri = val_ori
        if valx > maxImgX:
            maxImgX = valx
        if valy > maxImgY:
            maxImgY = valy
        if valx < minImgX:
            minImgX = valx
        if valy < minImgY:
            minImgY = valy
        
        tempmag.append(mg)
        tempori.append(val_ori)
        tempx.append(valx)
        tempxabs.append(abs(valx))
        tempy.append(valy)
        tempyabs.append(abs(valy))
        
    imgxedge.append(tempx)
    imgyedge.append(tempy)
    imgxedgeabs.append(tempxabs)
    imgyedgeabs.append(tempyabs)
    mag.append(tempmag)
    ori.append(tempori)

imageX = np.asarray(imgxedge)
cv2.imwrite("vericalDirection.png",imageX)
cv2.namedWindow('Vertical Edge Detection', cv2.WINDOW_NORMAL)
cv2.imshow('Vertical Edge Detection',imageX)
cv2.waitKey(0)
cv2.destroyAllWindows()

pos_edge_x = (imageX - minImgX) / (maxImgX - minImgX)
cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_x_dir', pos_edge_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

imageXabs = np.asarray(imgxedgeabs)
pos_edge_x = imageXabs / max(maxImgX, abs(minImgX))
cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_x_dir', pos_edge_x)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Sobel_x.jpg",pos_edge_x * 255)

imageY = np.asarray(imgyedge)
cv2.imwrite("horizontalDirection.png",imageY)
cv2.namedWindow('Horizontal Edge Detection', cv2.WINDOW_NORMAL)
cv2.imshow('Horizontal Edge Detection',imageY)
cv2.waitKey(0)
cv2.destroyAllWindows()

pos_edge_y = (imageY - minImgY) / (maxImgY - minImgY)
cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_y_dir', pos_edge_y)
cv2.waitKey(0)
cv2.destroyAllWindows()

imageYabs = np.asarray(imgyedgeabs)
pos_edge_y = imageYabs / max(maxImgY, abs(minImgY))
cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_y_dir', pos_edge_y)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Sobel_y.jpg",pos_edge_y * 255)

imgMag = np.asarray(mag)
imgMag /= maxMag
cv2.namedWindow('Magnitude', cv2.WINDOW_NORMAL)
cv2.imshow('Magnitude', imgMag)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("magnitude.jpg",imgMag * 255)

imgOri = np.asarray(ori)
imgOri /= maxOri
cv2.namedWindow('Direction', cv2.WINDOW_NORMAL)
cv2.imshow('Direction', imgOri)
cv2.waitKey(0)
cv2.destroyAllWindows()

