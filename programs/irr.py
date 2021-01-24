# import cv2
# import numpy as np
# import scipy.ndimage as ndimage
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt

# # img_raw = cv2.imread('sample.jpg', 1)
# # img_raw = cv2.imread('results/clustering.png', 1)
# img_raw = cv2.imread('results/labeling.png', 1)
# # img_raw = cv2.imread('800px-ColloidCrystal_10xBrightField_GlassInWater.jpg', 1)
# img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

# #画像の高さ, 幅を取得
# h, w = img.shape

# #画像の前処理(拡大)
# mag = 3
# img = cv2.resize(img, (w*mag, h*mag))

# #画像の前処理(ぼかし)
# img_blur = cv2.GaussianBlur(img,(5,5),0)

# #2値画像を取得
# ret,th = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# #モルフォロジー変換(膨張)
# kernel = np.ones((3,3),np.uint8)
# th = cv2.dilate(th,kernel,iterations = 1)

# #画像を保存
# cv2.imwrite('thresholds.png', th)

# #Fill Holes処理
# th_fill = ndimage.binary_fill_holes(th).astype(int) * 255
# cv2.imwrite('thresholds_fill.png', th_fill)

# #境界検出と描画
# cnt, __ = cv2.findContours(th_fill.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# img_raw = cv2.resize(img_raw, (w*mag, h*mag))
# img_cnt = cv2.drawContours(img_raw, cnt, -1, (0,255,255), 1)
# cv2.imwrite('cnt.png', img_cnt)

# #面積、円形度、等価直径を求める。
# Areas = []
# Circularities = []
# Eq_diameters = []

# for i in cnt:
# 	#面積(px*px)
# 	area = cv2.contourArea(i)
# 	Areas.append(area)

# 	#円形度
# 	arc = cv2.arcLength(i, True)
# 	circularity = 4 * np.pi * area / (arc * arc)
# 	Circularities.append(circularity)

# 	#等価直径(px)
# 	eq_diameter = np.sqrt(4*area/np.pi)
# 	Eq_diameters.append(eq_diameter)

# print(Circularities)


# import cv2
# import numpy as np
# import pymeanshift as pms
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from PIL import Image
# from collections import deque


# def trace(dummy,ys,xs):
# 	l,y,x,no,v = 0,ys,xs,dummy[ys][xs+1],5
# 	print('a')

	
# 	if ((x==xs)&(y==ys)&(l!=0)):
# 		dummy[y][x] = 255
# 		if (v==3):
# 			if ((dummy[y][x+1]!=no)&(dummy[y-1][x+1]==no)):
# 				x,y,l,vec = x+1,y,l+1,0
# 		if (v==4):
# 			if ((dummy[y-1][x+1]!=no)&(dummy[y-1][x]==no)):
# 				x,y,l,vec = x+1,y-1,l+np.sqrt(2),1
# 		if (v==5):
# 			if ((dummy[y-1][x]!=no)&(dummy[y-1][x-1]==no)):
# 				x,y,l,vec = x,y-1,l+1,2
# 		if (v==6):
# 			if ((dummy[y-1][x-1]!=no)&(dummy[y][x-1]==no)):
# 				x,y,l,vec = x-1,y-1,np.sqrt(2),3
# 		if (v==7):
# 			if ((dummy[y][x-1]!=no)&(dummy[y+1][x-1]==no)):
# 				x,y,l,vec = x-1,y,l+1,4
# 		if (v==0):
# 			if ((dummy[y+1][x-1]!=no)&(dummy[y+1][x]==no)):
# 				x,y,l,vec = x-1,y+1,l+np.sqrt(2),5
# 		if (v==1):
# 			if ((dummy[y+1][x]!=no)&(dummy[y+1][x+1]==no)):
# 				x,y,l,vec = x,y+1,l+1,6
# 		if (v==2):
# 			if ((dummy[y+1][x+1]!=no)&(dummy[y][x+1]==no)):
# 				x,y,l,vec = x+1,y+1,l+np.sqrt(2),7
				

# def length(dummy,label):
# 	h,w = dummy.shape[0],dummy.shape[1]

# 	for j in range(h):
# 		for i in range(w):
# 			if (dummy[j,i]==label):
# 				return trace(dummy,j-1,i)
# 	return 0


# img = cv2.imread('images/madslide11.jpg', cv2.IMREAD_COLOR)

# dummy = np.loadtxt('results/dummy.txt').astype(np.uint)
# label = np.loadtxt('results/label.txt').astype(np.uint)

# irr = length(dummy,1)

# print(irr)


