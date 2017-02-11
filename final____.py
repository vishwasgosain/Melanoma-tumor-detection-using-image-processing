import cv2
import numpy as np
from matplotlib import pyplot as plt

threshold = 0.8			#set the value of the threshold

inn = input('Enter the image name ')
img_rgb = cv2.imread(str(inn))

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

loc = [0]*80
template = [0]*80
w = [0]*80
h =  [0]*80
res = [0]*80

for i in range(1,43,1):
	template[i] = cv2.imread(str(i), 0)
	w[i],h[i] = template[i].shape[::-1]
	# = template[i].shape[::-1]
	res[i] = cv2.matchTemplate(img_gray,template[i],cv2.TM_CCOEFF_NORMED)
	loc[i] = np.where(res[i]>threshold)


"""
template = cv2.imread('gos.jpg',0)
w, h = template.shape[::-1]

w[i],h[i] = template.shape()

template1 = cv2.imread('gos2.jpg',0)
w1, h1 = template1.shape[::-1]

template2 = cv2.imread('gos3.jpg',0)
w2, h2 = template2.shape[::-1]

template3 = cv2.imread('gos4.jpg',0)
w3, h3 = template3.shape[::-1]

template4 = cv2.imread('gos5.jpg',0)
w4, h4 = template4.shape[::-1]



res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
res1 = cv2.matchTemplate(img_gray,template1,cv2.TM_CCOEFF_NORMED)
res2 = cv2.matchTemplate(img_gray,template2,cv2.TM_CCOEFF_NORMED)
res3 = cv2.matchTemplate(img_gray,template3,cv2.TM_CCOEFF_NORMED)
res4 = cv2.matchTemplate(img_gray,template4,cv2.TM_CCOEFF_NORMED)


threshold = 0.8
loc = np.where( res >= threshold)
loc1 = np.where( res1 >= threshold)
loc2 = np.where( res2 >= threshold)
loc3 = np.where( res3 >= threshold)
loc4 = np.where( res4 >= threshold)
"""
a =1
if a==1:

	

	for pt in zip(*loc[1][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print (" The mole is cancerous! ")
		a=0
	for pt in zip(*loc[2][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous!. ")
		a=0
	
	for pt in zip(*loc[3][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous. ")
		a=0
	for pt in zip(*loc[4][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print (" The mole is cancerous! ")
		a=0
	for pt in zip(*loc[5][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous!. ")
		a=0
	
	for pt in zip(*loc[6][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous. ")
		a=0
	for pt in zip(*loc[7][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print (" The mole is cancerous! ")
		a=0
	for pt in zip(*loc[8][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous!. ")
		a=0
	
	for pt in zip(*loc[9][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous. ")
		a=0
	for pt in zip(*loc[10][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print (" The mole is cancerous! ")
		a=0
	for pt in zip(*loc[11][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous!. ")
		a=0
	
	for pt in zip(*loc[12][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous. ")
		a=0
	for pt in zip(*loc[13][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print (" The mole is cancerous! ")
		a=0
	for pt in zip(*loc[14][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous!. ")
		a=0
	
	for pt in zip(*loc[15][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous. ")
		a=0
	for pt in zip(*loc[16][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print (" The mole is cancerous! ")
		a=0
	for pt in zip(*loc[17][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous!. ")
		a=0
	
	for pt in zip(*loc[18][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	for pt in zip(*loc[19][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print (" The mole is cancerous! ")
		a=0
	for pt in zip(*loc[20][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous!. ")
		a=0
	
	for pt in zip(*loc[21][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	for pt in zip(*loc[22][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print (" The mole is cancerous! ")
		a=0
	for pt in zip(*loc[23][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous!. ")
		a=0
	
	for pt in zip(*loc[24][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous. ")
		a=0
	for pt in zip(*loc[25][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print (" The mole is cancerous! ")
		a=0
	for pt in zip(*loc[26][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous!. ")
		a=0
	
	for pt in zip(*loc[27][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous. ")
		a=0
	for pt in zip(*loc[28][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print (" The mole is cancerous! ")
		a=0
	for pt in zip(*loc[29][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous!. ")
		a=0
	
	for pt in zip(*loc[30][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is cancerous. ")
		a=0
	
	
	
	for pt in zip(*loc[31][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	
	for pt in zip(*loc[32][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	
	for pt in zip(*loc[33][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	
	for pt in zip(*loc[34][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	
	for pt in zip(*loc[35][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	
	for pt in zip(*loc[36][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	
	for pt in zip(*loc[37][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	
	for pt in zip(*loc[38][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	
	for pt in zip(*loc[39][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	
	for pt in zip(*loc[40][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	
	for pt in zip(*loc[41][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	
	for pt in zip(*loc[42][::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w[0], pt[1] + h[0]), (0,0,255), 2)
		print ("Mole is not cancerous. ")
		a=0
	
				
				
re = cv2.resize(img_rgb, (500, 250))
cv2.imshow('rupee',re)

cv2.waitKey(0)
cv2.destroyAllWindows()
