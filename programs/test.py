import cv2
import numpy as np
import pymeanshift as pms
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from collections import deque

def image(_org,_img):
  global org,img,h,w,c
  global bo,go,ro,al
  
  org,img = _org,_img
  h,w,c = img.shape
  bo,go,ro = cv2.split(org)
  al = 0.55

def shortcut():
  global dummy,label
  dummy = np.loadtxt('results/dummy.txt').astype(np.uint)
  label = np.loadtxt('results/label.txt').astype(np.uint)

def detection():
  bl,gl,rl = cv2.split(org)
  bf,gf,rf = cv2.split(org)
  _lnd,_fld = np.full((h,w), 255),np.full((h,w), 255)

  lab,hsv = cv2.cvtColor(img,cv2.COLOR_BGR2Lab),cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
 
  for l in range(1,label+1):
    idx = np.where(dummy==l)

    Lp,ap,bp = np.split(lab[idx],3,axis=1)
    hp,sp,vp = np.split(hsv[idx],3,axis=1)
    edge = cv2.imread('results/edge/canny.png', cv2.IMREAD_GRAYSCALE)[idx]

    print(edge.shape)

    landslide = (Lp<180)&(ap>128)&(sp>70)&(edge>100)
    _flooded = (Lp>135)&(ap>128)&(sp<=70)
    flooded = _flooded&(~landslide)

    if (np.count_nonzero(landslide)>np.count_nonzero(~landslide)):
      bl[idx],gl[idx],rl[idx] = (bo[idx]),(go[idx]),(ro[idx]*al+250*(1-al))
      _lnd[idx] = 0

    if (np.count_nonzero(flooded)>np.count_nonzero(~flooded)):
      bf[idx],gf[idx],rf[idx] = (bo[idx]),(go[idx]*al+250*(1-al)),(ro[idx]*al+250*(1-al))
      _fld[idx] = 0
 
  lnd,fld = np.dstack((np.dstack((bl,gl)),rl)),np.dstack((np.dstack((bf,gf)),rf))

  cv2.imwrite('results/landslide.png', lnd)
  cv2.imwrite('results/flooded.png', fld)
  cv2.imwrite('results/_landslide.png', _lnd)
  cv2.imwrite('results/_flooded.png', _fld)

  return _lnd,_fld


# 入力画像読込
img = cv2.imread('results/meanshift500.png', cv2.IMREAD_COLOR)

# オリジナル画像
org = cv2.imread('results/original.png', cv2.IMREAD_COLOR)

# 画像データ
image(org,img)

# ラベリングデータ
shortcut()

# 災害領域検出
#   - 斜面崩壊・瓦礫でピクセル穴がある　
#   - 浸水検出結果で緑色のノイズがある
lnd,fld = detection()
