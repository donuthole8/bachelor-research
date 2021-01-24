import cv2
import numpy as np
import pymeanshift as pms
from PIL import Image
from collections import deque

def meanshift(rgb_img, spatial_radius, range_radius, min_density):
    lab_img, labels, num_seg = pms.segment(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2Lab), spatial_radius, range_radius,min_density)
    new_rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)
    return (new_rgb_img, labels, num_seg)

def contrastize(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    new_v = clahe.apply(v)

    hsv_clahe = cv2.merge((h, s, new_v))
    new_rgb_img = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    return new_rgb_img

def cluster(contrast_img):
    img = Image.fromarray(contrast_img)
    img_q = img.quantize(colors=128, method=0, dither=1)
    a = np.asarray(img_q)
    return np.stack([a]*3, axis=2)

def approximation(pix1, pix2):
    dif = abs(pix1 - pix2)
    # dv = 10
    dv = 15
    return (dif < dv).all()

def neighbours(idx, lim):
    w, h, _ = lim
    i, j = idx
    return sorted([
            (i + n, j + m)
            for n in range(-1, 2)
            for m in range(-1, 2)
            if not (n == 0 and m == 0) and i + n >= 0 and j + m >= 0 and i + n < w and j + m < h
            ])


def relabel(dummy, src, idx, labels, label):
    q = deque([idx])
    while len(q) > 0:
        idx = q.popleft()
        labels[idx] = (label * 5, label * 10, label * 30)
        dummy[idx] = label
        ns = neighbours(idx, src.shape)
        q.extendleft(n for n in ns if approximation(src[n], src[idx]) and dummy[n] == 0)


def label(filename):
    src = cv2.imread(filename)
    if src is None:
        raise Exception
    # (img_shifted, labels, num_seg) = meanshift(src, 12, 3, 200)
    img_shifted = src
    img_cont = contrastize(img_shifted)
    img_cluster = cluster(img_cont)

    h, w, c = src.shape
    dummy = np.zeros((h, w), dtype=int)
    labels = np.zeros((h, w, c), dtype=int)
    label = 1

    it = np.nditer(img_cluster, flags=['multi_index'], op_axes=[[0, 1]])
    for n in it:
        if dummy[it.multi_index] == 0:
            relabel(dummy, img_cluster, it.multi_index,labels, label)
            label += 1

    print('label number :',label)
    cv2.imwrite('result.png', labels.astype(np.uint8))

if __name__ == '__main__':
    label('results/meanshift500.png')
    # label('results/meanshift1920.png')







# import cv2
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np
# import pymeanshift as pms
# import os
# import sys
# from PIL import Image
# # sys.setrecursionlimit(8000) # 200 x 113 pix
# sys.setrecursionlimit(30000)  # 500 x 281 pix
# plt.gray()


# def image(_org,_img):
#   global org,img,h,w,c
#   global bo,go,ro,al
  
#   org,img = _org,_img
#   h,w,c = img.shape
#   bo,go,ro = cv2.split(org)
#   al = 0.55

# def meanshift(spatial_radius,range_radius,min_density):
#   global img
#   (img,labels,num) = pms.segment(cv2.cvtColor(img,cv2.COLOR_BGR2Lab),spatial_radius,range_radius,min_density)
#   img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
#   cv2.imwrite('results/meanshift.png',img)

def shortcut1():
  global img
  img = cv2.imread('results/meanshift.png', cv2.IMREAD_COLOR)
def shortcut2():
  global img
  img = cv2.imread('results/meanshift500.png', cv2.IMREAD_COLOR)
def shortcut3():
  global img
  img = cv2.imread('results/meanshift1920.png', cv2.IMREAD_COLOR)

# def contrast():
#   global img
#   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#   h,s,v = cv2.split(hsv)

#   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
#   result = clahe.apply(v)

#   hsv = cv2.merge((h,s,result))
#   img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#   cv2.imwrite("results/contrast.png", img)

# def quantization():
#   global img
#   img = img // 4 * 4
#   cv2.imwrite('results/quantization.png', img)

# def division():
#   global block
#   hs,ws,cnt = 113,200,1

#   if ((h>hs)&(w>ws)):
#     for j in range(0,h,hs):
#       for i in range(0,w,ws):
#         if (((j+hs)<h)&((i+ws)<w)):
#           div = np.zeros((hs,ws,c),dtype=int)
#           div[0:hs,0:ws] = img[j:j+hs,i:i+ws]
#         else:
#           if (((i+ws)>w)&((j+hs)>h)):
#             div = np.zeros((h-j,w-i,c),dtype=int)
#             div[0:h-j,0:w-i] = img[j:h,i:w]
#           elif ((j+hs)>h):
#             div = np.zeros((h-j,ws,c),dtype=int)
#             div[0:h-j,0:ws] = img[j:h,i:i+ws]
#           else:
#             div = np.zeros((hs,w-i,c),dtype=int)
#             div[0:hs,0:w-i] = img[j:j+hs,i:w]
#         cv2.imwrite('results/division/division{}.png'.format(cnt),div)
#         cnt += 1
#   else:
#     cv2.imwrite('results/division/division1.png',img)
#   block = cnt
#   print('block number :',block)

# def clustering():
#   global img
#   im = Image.open('results/contrast.png')
#   im_q = im.quantize(colors=128, method=0, dither=1)
#   im_q.save('results/clustering.png')
#   img = cv2.imread('results/clustering.png', cv2.IMREAD_COLOR)

# def approximation(pix1,pix2):
#   dif = abs(pix1.astype(np.int)-pix2.astype(np.int))
#   d1,d2,d3 = dif[0],dif[1],dif[2]
#   dv = 20
#   # print(dif)
#   if ((d1<dv)&(d2<dv)&(d3<dv)):
#     return True
#   else:
#     return False

# def relabeling(j,i,pix):
#   global _dm,dm,_la,la,label
#   # 同色領域にラベル付与
#   if ((j>=1)&(i>=1)&(j<_la.shape[0]-1)&(i<_la.shape[1]-1)):
#     _dm[j,i] = label
#     la[j-1,i-1] = label*5,label*10,label*30
#     # 8近傍画素で注目画素と同じインデックス取得
#     # idx = np.where(img[j-1:j+2,i-1:i+2]==pix)
    
#     # すっきりさせたい
#     # # 4近傍
#     # if (approximation(pix,_la[j,i-1])&(_dm[j,i-1]==0)):
#     #   relabeling(j,i-1,pix)
#     # if (approximation(pix,_la[j-1,i])&(_dm[j-1,i]==0)):
#     #   relabeling(j-1,i,pix)
#     # if (approximation(pix,_la[j+1,i])&(_dm[j+1,i]==0)):
#     #   relabeling(j+1,i,pix)
#     # if (approximation(pix,_la[j,i+1])&(_dm[j,i+1]==0)):
#     #   relabeling(j,i+1,pix)
    
#     # # 4近傍（類似色統合しない）
#     # if ((pix==_la[j,i-1]).all()&(_dm[j,i-1]==0)):
#     #   relabeling(j,i-1,pix)
#     # if ((pix==_la[j-1,i]).all()&(_dm[j-1,i]==0)):
#     #   relabeling(j-1,i,pix)
#     # if ((pix==_la[j+1,i]).all()&(_dm[j+1,i]==0)):
#     #   relabeling(j+1,i,pix)
#     # if ((pix==_la[j,i+1]).all()&(_dm[j,i+1]==0)):
#     #   relabeling(j,i+1,pix)

#     # if (approximation(pix,_la[j,i-1])):
#     #   relabeling(j,i-1,pix)
#     # if (approximation(pix,_la[j-1,i])):
#     #   relabeling(j-1,i,pix)
#     # if (approximation(pix,_la[j+1,i])):
#     #   relabeling(j+1,i,pix)
#     # if (approximation(pix,_la[j,i+1])):
#     #   relabeling(j,i+1,pix)

#     # 8近傍
#     if (approximation(pix,_la[j-1,i-1])&(_dm[j-1,i-1]==0)):
#       relabeling(j-1,i-1,pix)
#     if (approximation(pix,_la[j,i-1])&(_dm[j,i-1]==0)):
#       relabeling(j,i-1,pix)
#     if (approximation(pix,_la[j+1,i-1])&(_dm[j+1,i-1]==0)):
#       relabeling(j+1,i-1,pix)
    
#     if (approximation(pix,_la[j-1,i])&(_dm[j-1,i]==0)):
#       relabeling(j-1,i,pix)
#     if (approximation(pix,_la[j+1,i])&(_dm[j+1,i]==0)):
#       relabeling(j+1,i,pix)
    
#     if (approximation(pix,_la[j-1,i+1])&(_dm[j-1,i+1]==0)):
#       relabeling(j-1,i+1,pix)
#     if (approximation(pix,_la[j,i+1])&(_dm[j,i+1]==0)):
#       relabeling(j,i+1,pix)
#     if (approximation(pix,_la[j+1,i+1])&(_dm[j+1,i+1]==0)):
#       relabeling(j+1,i+1,pix)

#     # 8近傍（類似色統合しない）
#     # if ((pix==_la[j-1,i-1]).all()&(_dm[j-1,i-1]==0)):
#     #   relabeling(j-1,i-1,pix)
#     # if ((pix==_la[j,i-1]).all()&(_dm[j,i-1]==0)):
#     #   relabeling(j,i-1,pix)
#     # if ((pix==_la[j+1,i-1]).all()&(_dm[j+1,i-1]==0)):
#     #   relabeling(j+1,i-1,pix)

#     # if ((pix==_la[j-1,i]).all()&(_dm[j-1,i]==0)):
#     #   relabeling(j-1,i,pix)
#     # if ((pix==_la[j+1,i]).all()&(_dm[j+1,i]==0)):
#     #   relabeling(j+1,i,pix)

#     # if ((pix==_la[j-1,i+1]).all()&(_dm[j-1,i+1]==0)):
#     #   relabeling(j-1,i+1,pix)
#     # if ((pix==_la[j,i+1]).all()&(_dm[j,i+1]==0)):
#     #   relabeling(j,i+1,pix)
#     # if ((pix==_la[j+1,i+1]).all()&(_dm[j+1,i+1]==0)):
#     #   relabeling(j+1,i+1,pix)

# def _labeling(div):
#   global _dm,dm,_la,la,label
#   dh,dw = div.shape[0],div.shape[1]
#   _dm,_la,la = np.zeros((dh+2,dw+2),dtype=int),np.zeros((dh+2,dw+2,c),dtype=int),np.zeros((dh,dw,c),dtype=int)

#   _la[1:dh+1,1:dw+1] = div
#   # label = 1

#   # for文でのラスタスキャンを用いない方が好ましい
#   for j in range(1,dh+1):
#     for i in range(1,dw+1):
#       # ダミー配列にラベル付けされている画素ならば処理を行わない
#       if (_dm[j,i]==0):
#         # 注目画素値取得
#         pix = div[j-1,i-1]
#         # ラベリング
#         relabeling(j,i,pix)
#         # ラベル更新
#         label += 1

#   print('label number :',label)

#   dm = np.zeros((dh,dw,c),dtype=int)
#   dm = _dm[1:dh+1,1:dw+1]

#   la = la.astype(np.uint8)
  
#   return dm,la

# def labeling():
#   global label
#   label,n = 1,block

#   for i in range(1,n+1):
#     div = cv2.imread('results/division/division{}.png'.format(i),cv2.IMREAD_COLOR)
#     d,l = _labeling(div)
#     np.savetxt('results/dummy/dummy{}.txt'.format(i),d,fmt='%d')
#     cv2.imwrite('results/dummy/dummy{}.png'.format(i),d)
#     cv2.imwrite('results/labeling/labeling{}.png'.format(i),l)





def texture():
  mpl.rc('image', cmap='jet')
  pass
  kernel_size = 5
  levels = 8
  symmetric = False
  normed = True
  # オリジナル画像でのテクスチャ解析
  dst = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
  # 前処理済み画像でのテクスチャ解析
  # dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # binarize
  dst_bin = dst//(256//levels) # [0:255]->[0:7]

  # calc_glcm             
  h,w = dst.shape
  glcm = np.zeros((h,w,levels,levels), dtype=np.uint8)
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  dst_bin_r = np.append(dst_bin[:,1:], dst_bin[:,-1:], axis=1)
  for i in range(levels):
    for j in range(levels):
      mask = (dst_bin==i) & (dst_bin_r==j)
      mask = mask.astype(np.uint8)
      glcm[:,:,i,j] = cv2.filter2D(mask, -1, kernel)
  glcm = glcm.astype(np.float32)
  if symmetric:
    glcm += glcm[:,:,::-1, ::-1]
  if normed:
    glcm = glcm/glcm.sum(axis=(2,3), keepdims=True)
  # martrix axis
  axis = np.arange(levels, dtype=np.float32)+1
  w = axis.reshape(1,1,-1,1)
  x = np.repeat(axis.reshape(1,-1), levels, axis=0)
  y = np.repeat(axis.reshape(-1,1), levels, axis=1)
  # GLCM contrast
  glcm_contrast = np.sum(glcm*(x-y)**2, axis=(2,3))
  # GLCM dissimilarity
  glcm_dissimilarity = np.sum(glcm*np.abs(x-y), axis=(2,3))
  # GLCM homogeneity
  glcm_homogeneity = np.sum(glcm/(1.0+(x-y)**2), axis=(2,3))
  # GLCM energy & ASM
  glcm_asm = np.sum(glcm**2, axis=(2,3))
  # GLCM entropy
  ks = 5 # kernel_size
  pnorm = glcm / np.sum(glcm, axis=(2,3), keepdims=True) + 1./ks**2
  glcm_entropy = np.sum(-pnorm * np.log(pnorm), axis=(2,3))
  # GLCM mean
  glcm_mean = np.mean(glcm*w, axis=(2,3))
  # GLCM std
  glcm_std = np.std(glcm*w, axis=(2,3))
  # GLCM energy
  glcm_energy = np.sqrt(glcm_asm)
  # GLCM max
  glcm_max = np.max(glcm, axis=(2,3))
  
  # plot
  plt.figure(figsize=(10,4.5))

  outs =[dst, glcm_mean, glcm_std,
    glcm_contrast, glcm_dissimilarity, glcm_homogeneity,
    glcm_asm, glcm_energy, glcm_max,
    glcm_entropy]
  titles = ['original','mean','std','contrast','dissimilarity','homogeneity','ASM','energy','max','entropy']
  for i in range(10):
    plt.imsave('results/tex/' + titles[i] + '.png', outs[i])
  
  return

def edge():
  # オリジナル画像に対してエッジ抽出
  dst = cv2.cvtColor(org.copy(), cv2.COLOR_BGR2GRAY)
  # 前処理済み画像に対してエッジ抽出
  # dst = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

  canny = cv2.Canny(dst,100,150,5)
  sobelx = cv2.Sobel(dst,cv2.CV_64F,1,0,ksize=5)
  sobely = cv2.Sobel(dst,cv2.CV_64F,0,1,ksize=5)
  laplacian = cv2.Laplacian(dst,cv2.CV_64F)
  
  canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
  sobel = np.sqrt(sobelx**2 + sobely**2)
  # laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

  plt.imsave('results/edge/canny.png', canny)
  plt.imsave('results/edge/sobelx.png', sobelx)
  plt.imsave('results/edge/sobely.png', sobely)
  plt.imsave('results/edge/sobel.png', sobel)
  plt.imsave('results/edge/laplacian.png', laplacian)

  return

def detection():
  bl,gl,rl = cv2.split(org)
  bf,gf,rf = cv2.split(org)
  _lnd,_fld = np.full((h,w), 255),np.full((h,w), 255)

  # GSI粒度指数：微妙だけど領域処理では活きてくるかも
  # _bo,_go,_ro = cv2.split(org)
  # _ro[np.where((_ro+_bo+_go)==0)] = 255
  # gsi = ((_ro-_bo))/(_ro+_bo+_go)
  # gsi = (gsi*100).astype(np.uint8)
  # idx = np.where(gsi>0.3)

  # S彩度：かなりいい指標、単体で斜面崩壊を検出できそう
  hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  hh,s,v = cv2.split(hsv)
  # idx = np.where(s>95)  # 斜面崩壊検出
  # idx = np.where(s<75)  # 浸水検出

  # z_saji
  _b,_g,_r = cv2.split(img)
  _r[np.where((_g+_b)==0)] = 255
  z = (_g-_r)/(_g+_b)
  z = (z*100).astype(np.uint8)
  idx = np.where(z>50)

  b,g,r = cv2.split(org)
  b[idx],g[idx],r[idx] = (bo[idx]),(go[idx]*al+250*(1-al)),(ro[idx]*al+250*(1-al))

  res = np.dstack((np.dstack((b,g)),r))

  # cv2.imwrite('gsi.jpg',gsi)
  cv2.imwrite('saji.jpg',z)
  cv2.imwrite('result.jpg',res)

  # 領域単位での投票処理
  for l in range(1,label):
    idx = np.where(dm==l)

    lab,hsv = cv2.cvtColor(img,cv2.COLOR_BGR2Lab),cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    Lp,ap,bp = np.split(lab[idx],3,axis=1)
    hp,sp,vp = np.split(hsv[idx],3,axis=1)
    dis = cv2.imread('results/tex/dissimilarity.png', cv2.IMREAD_GRAYSCALE)[idx]
    edge = cv2.imread('results/edge/canny.png', cv2.IMREAD_GRAYSCALE)[idx]

    ep = (np.count_nonzero(edge==0))/(np.count_nonzero(dm==l))
    print(ep)
    

    # _bo,_go,_ro = np.split(org[idx],3,axis=1)
    # _ro[np.where((_ro+_bo+_go)==0)] = 1
    # gsi = ((_ro-_bo))/(_ro+_bo+_go)


    landslide = (Lp<180)&(ap>128)&(sp>70)
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

def rejection():
  bs,gs,rs = cv2.split(org)
  bv,gv,rv = cv2.split(org)
  br,gr,rr = cv2.split(org)
  bb,gb,rb = cv2.split(org)
  _sky,_veg = np.full((h,w), 255),np.full((h,w), 255)
  _rbl,_bld = np.full((h,w), 255),np.full((h,w), 255)
  mask = np.full((h,w), 255)
  
  # 領域単位での投票処理
  for l in range(1,label):
    idx = np.where(dm==l)

    lab,hsv = cv2.cvtColor(img,cv2.COLOR_BGR2Lab),cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    Lp,ap,bp = np.split(lab[idx],3,axis=1)
    hp,sp,vp = np.split(hsv[idx],3,axis=1)
    _bo,_go,_ro = np.split(org[idx],3,axis=1)
  
    # # 斜面崩壊
    # landslide = (Lp<180)&(ap>128)&(sp>70)
    # # 浸水
    # _flooded = (Lp>135)&(ap>128)&(sp<=70)
    # flooded = _flooded&(~landslide)
    # 空
    sky = (Lp>=200)&(bp<=135)&(ap>=125)
    # 植生
    _vegitation = ((ap<130)|(bp<130))&(hp>20)
    vegitation = _vegitation&(~sky)
    # 瓦礫
    _dis = cv2.imread('results/tex/dissimilarity.png', cv2.IMREAD_GRAYSCALE)
    dis = _dis[idx]
    _rubble = ((dis>120)&(ap>120)&(Lp<140))
    # _landslide,_flooded = landslide[idx],flooded[idx]
    # rubble = _rubble&(~((_landslide)&(_flooded)))
    rubble = _rubble
    # 建物

    # imgray = cv2.cvtColor(labeling, cv2.COLOR_BGR2GRAY)
    # im_gauss = cv2.GaussianBlur(imgray, (5, 5), 0)
    # ret, thresh = cv2.threshold(im_gauss, 127, 255, 0)

    # cv2.imwrite('thresh.png',thresh)
    # # get contours
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # contours_cirles = []

    # # check if contour is of circular shape
    # for con in contours:
    #   perimeter = cv2.arcLength(con, True)
    #   area = cv2.contourArea(con)
    #   if perimeter == 0:
    #     break
    #   circularity = 4*np.pi*(area/(perimeter*perimeter))
    #   print(circularity)
    #   if 0.7 < circularity < 1.2:
    #     contours_cirles.append(con)

    # print(idx)

    # L,S = 0,np.count_nonzero(dm==l)
    # irr = L**2/(4*np.pi*S)

    _ro[np.where((_ro+_bo+_go)==0)] = 1
    gsi = ((_ro-_bo))/(_ro+_bo+_go)
    building = gsi<0.15

    if (np.count_nonzero(sky)>np.count_nonzero(~sky)):
      bs[idx],gs[idx],rs[idx] = (bo[idx]*al+250*(1-al)),(go[idx]),(ro[idx]*al+100*(1-al))
      _sky[idx],mask[idx] = 0,0
    if (np.count_nonzero(vegitation)>np.count_nonzero(~vegitation)):
      bv[idx],gv[idx],rv[idx] = (bo[idx]*al+100*(1-al)),(go[idx]*al+200*(1-al)),(ro[idx])
      _veg[idx],mask[idx] = 0,0
    if (np.count_nonzero(rubble)>np.count_nonzero(~rubble)):
      br[idx],gr[idx],rr[idx] = (bo[idx]),(go[idx]),(ro[idx]*al+250*(1-al))
      _rbl[idx],mask[idx] = 0,0
    if (np.count_nonzero(building)>np.count_nonzero(~building)):
      bb[idx],gb[idx],rb[idx] = (bo[idx]),(go[idx]),(ro[idx]*al+250*(1-al))
      _bld[idx],mask[idx] = 0,0

  # print(np.where(dm==1))

  # 空領域除去
  # idx = np.where(sky)
  # b[idx],g[idx],r[idx] = (b[idx]*al+250*(1-al)),(g[idx]),(r[idx]*al+100*(1-al))
  # mask[idx] = 0
  # # 植生領域除去
  # idx = np.where(vegitation)
  # b[idx],g[idx],r[idx] = (b[idx]*al+100*(1-al)),(g[idx]*al+200*(1-al)),r[idx]
  # mask[idx] = 0
  # # 瓦礫除去
  # idx = np.where(rubble)
  # # マスク画像作成
  # b[idx],g[idx],r[idx] = (b[idx]*al+250*(1-al)),(g[idx]),(r[idx]*al+150*(1-al))
  # mask[idx] = 0
  # 建物除去
  # idx = np.where(building)
  # mask[idx] = 0
  # b[idx],g[idx],r[idx] = (b[idx]*al+250*(1-al)),(g[idx]),(r[idx]*al+150*(1-al))
  # # mask[idx] = 0

  sky,veg = np.dstack((np.dstack((bs,gs)),rs)),np.dstack((np.dstack((bv,gv)),rv))
  rbl,bld = np.dstack((np.dstack((br,gr)),rr)),np.dstack((np.dstack((bb,gb)),rb))

  cv2.imwrite('results/sky.png', sky)
  cv2.imwrite('results/vegitation.png', veg)
  cv2.imwrite('results/rubble.png', rbl)
  cv2.imwrite('results/building.png', bld)
  cv2.imwrite('results/_sky.png', _sky)
  cv2.imwrite('results/_vegitation.png', _veg)
  cv2.imwrite('results/_rubble.png', _rbl)
  cv2.imwrite('results/_building.png', _bld)
  cv2.imwrite('results/mask.png', mask)

  return mask

def integration(mask,landslide,flooded):
  b,g,r = cv2.split(org)
  lnd,fld = np.full((h,w), 255),np.full((h,w), 255)

  # 斜面崩壊
  idx = np.where(landslide==0)
  b[idx],g[idx],r[idx] = (bo[idx]),(go[idx]),(ro[idx]*al+250*(1-al))
  lnd[idx] = 0

  # 浸水
  idx = np.where(flooded==0)
  b[idx],g[idx],r[idx] = (bo[idx]),(go[idx]*al+250*(1-al)),(ro[idx]*al+250*(1-al))
  fld[idx] = 0

  # マスク処理（空・植生・瓦礫・建物）
  idx = np.where(mask==0)
  b[idx],g[idx],r[idx] = bo[idx],go[idx],ro[idx]
  lnd[idx],fld[idx] = 255,255
  
  res = np.dstack((np.dstack((b,g)),r))

  cv2.imwrite('results/result.png', res)
  return lnd,fld

def evaluation(lnd,fld):
  # ans_lnd = cv2.imread('images/landslide_answer_500.png', cv2.IMREAD_GRAYSCALE)
  ans_lnd = cv2.imread('images/landslide_answer_500.jpg', cv2.IMREAD_GRAYSCALE)
  ans_fld = cv2.imread('images/flooded_answer_500.jpg', cv2.IMREAD_GRAYSCALE)
  
  tp = np.count_nonzero((lnd==0)&(ans_lnd==255))
  fp = np.count_nonzero((lnd==0)&(ans_lnd!=255))
  fn = np.count_nonzero((lnd!=0)&(ans_lnd==255))
  # tn = np.count_nonzero((lnd!=0)&(ans_lnd!=255))
  print('landslide evaluation')
  recall = tp/(tp+fn)
  print('\trecall :','{:.3g}'.format(recall))
  precicsion = tp/(tp+fp)
  print('\tprecicsion :','{:.3g}'.format(precicsion))
  f1 = 2*(recall*precicsion)/(recall+precicsion)
  print('\tf1-score :','{:.3g}'.format(f1))

  tp = np.count_nonzero((fld==0)&(ans_fld==255))
  fp = np.count_nonzero((fld==0)&(ans_fld!=255))
  fn = np.count_nonzero((fld!=0)&(ans_fld==255))
  # tn = np.count_nonzero((fld!=0)&(ans_fld!=255))
  print('flooded evaluation')
  recall = tp/(tp+fn)
  print('\trecall :','{:.3g}'.format(recall))
  precicsion = tp/(tp+fp)
  print('\tprecicsion :','{:.3g}'.format(precicsion))
  f1 = 2*(recall*precicsion)/(recall+precicsion)
  print('\tf1-score :','{:.3g}'.format(f1))
