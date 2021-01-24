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

def meanshift(spatial_radius,range_radius,min_density):
  global img
  lab_img, labels, num_seg = pms.segment(cv2.cvtColor(img, cv2.COLOR_BGR2Lab), spatial_radius, range_radius,min_density)
  new_rgb_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)

  img = new_rgb_img
  cv2.imwrite('results/meanshift.png',img)

def shortcut1():
  global img
  img = cv2.imread('results/meanshift.png', cv2.IMREAD_COLOR)
def shortcut2():
  global img
  img = cv2.imread('results/meanshift500.png', cv2.IMREAD_COLOR)
def shortcut3():
  global img
  img = cv2.imread('results/meanshift1920.png', cv2.IMREAD_COLOR)

def equalization():
  global img
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv_img)

  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
  new_v = clahe.apply(v)

  hsv_clahe = cv2.merge((h, s, new_v))
  new_rgb_img = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

  img = new_rgb_img
  cv2.imwrite('results/equalization.png',img)

def clustering():
  global img
  img = Image.fromarray(img)
  img_q = img.quantize(colors=128, method=0, dither=1)
  img_q.save('results/clustering.png')
  img = cv2.imread('results/clustering.png', cv2.IMREAD_COLOR)

  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  cv2.imwrite('results/clustering.png',img)

def approximation(pix1, pix2):
  dif = abs(pix1 - pix2)
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

def relabeling(dummy, src, idx, labels, label):
  q = deque([idx])
  while len(q) > 0:
    idx = q.popleft()
    labels[idx] = (label * 5, label * 10, label * 30)
    dummy[idx] = label
    ns = neighbours(idx, src.shape)
    q.extendleft(n for n in ns if approximation(src[n], src[idx]) and dummy[n] == 0)

def labeling():
  global img,dummy,label

  dummy = np.zeros((h, w), dtype=int)
  labels = np.zeros((h, w, c), dtype=int)
  label = 1

  it = np.nditer(img, flags=['multi_index'], op_axes=[[0, 1]])
  for n in it:
    if dummy[it.multi_index] == 0:
      relabeling(dummy, img, it.multi_index,labels, label)
      label += 1

  print('label number :',label)

  np.savetxt('results/dummy.txt',dummy.astype(np.uint8),fmt='%d')
  with open('results/label.txt', 'w') as f:
    print(label, file=f)
  cv2.imwrite('results/labeling.png', labels.astype(np.uint8))
  cv2.imwrite('results/dummy.png', dummy.astype(np.uint8))

def shortcut_2():
  global dummy,label
  dummy = np.loadtxt('results/dummy.txt').astype(np.uint)
  label = np.loadtxt('results/label.txt').astype(np.uint)
  # print(dummy,label)

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
  # plt.figure(figsize=(10,4.5))

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
  sobel = np.sqrt(sobelx**2 + sobely**2)
  laplacian = cv2.Laplacian(dst,cv2.CV_64F)
  
  cv2.imwrite('results/edge/canny.png', canny)
  cv2.imwrite('results/edge/sobelx.png', sobelx)
  cv2.imwrite('results/edge/sobely.png', sobely)
  cv2.imwrite('results/edge/sobel.png', sobel)
  cv2.imwrite('results/edge/laplacian.png', laplacian)

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

  # # S彩度：かなりいい指標、単体で斜面崩壊を検出できそう
  # hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  # hh,s,v = cv2.split(hsv)
  # # idx = np.where(s>95)  # 斜面崩壊検出
  # # idx = np.where(s<75)  # 浸水検出

  # # z_saji
  # _b,_g,_r = cv2.split(img)
  # _r[np.where((_g+_b)==0)] = 255
  # z = (_g-_r)/(_g+_b)
  # z = (z*100).astype(np.uint8)
  # idx = np.where(z>50)

  # b,g,r = cv2.split(org)
  # b[idx],g[idx],r[idx] = (bo[idx]),(go[idx]*al+250*(1-al)),(ro[idx]*al+250*(1-al))

  # res = np.dstack((np.dstack((b,g)),r))

  # # cv2.imwrite('gsi.png',gsi)
  # cv2.imwrite('saji.png',z)
  # cv2.imwrite('result.png',res)

  lab,hsv = cv2.cvtColor(img,cv2.COLOR_BGR2Lab),cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  _dis = cv2.imread('results/tex/dissimilarity.png', cv2.IMREAD_COLOR)
  _edge = cv2.imread('results/edge/canny.png', cv2.IMREAD_COLOR)

  # 領域単位での投票処理
  for l in range(1,label):
    idx = np.where(dummy==l)

    Lp,ap,bp = np.split(lab[idx],3,axis=1)
    hp,sp,vp = np.split(hsv[idx],3,axis=1)
    edge = np.split(_edge[idx],3,axis=1)[0]

    # エッジ抽出率（エッジ画像中の白色領域）
    ep = (np.count_nonzero(edge==255))/(np.count_nonzero(dummy==l))
    # print(ep)

    # 多めに検出（誤検出含んでよし）しといて除去でたくさん弾くのあり

    landslide = (Lp<180)&(ap>128)&(sp>60)
    # _flooded = (Lp>135)&(ap>128)&(sp<=70)&(ep<0.2)
    # flooded = (Lp>135)&(ap>128)&(sp<=80)&(ep<0.2)
    flooded = (Lp>135)&(ap>128)&(sp<=80)&(ep<0.2)
    # flooded = _flooded&(~landslide)

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
  _dis = cv2.imread('results/tex/dissimilarity.png', cv2.IMREAD_COLOR)
  _edge = cv2.imread('results/edge/canny.png', cv2.IMREAD_COLOR)

  # 領域単位での投票処理
  for l in range(1,label):
    idx = np.where(dummy==l)

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
    dis = np.split(_dis[idx],3,axis=1)[0]
    edge = np.split(_edge[idx],3,axis=1)[0]
    dp = (np.count_nonzero(dis>200))/(np.count_nonzero(dummy==l))
    ep = (np.count_nonzero(edge==255))/(np.count_nonzero(dummy==l))
    
    # _rubble = ((dis>120)&(ap>120)&(Lp<140))

    # _rubble = ((dis>120)&(ap>120)&(Lp<140))
    # print(_rubble)

    # _landslide,_flooded = landslide[idx],flooded[idx]
    # rubble = _rubble&(~((_landslide)&(_flooded)))
    # rubble = (ep>0.35)
    rubble = (dp>0.7)&(ep>0.3)
    
    # 建物
    # L,S = 0,np.count_nonzero(dm==l)
    # irr = L**2/(4*np.pi*S)
    _ro[np.where((_ro+_bo+_go)==0)] = 1
    gsi = ((_ro-_bo))/(_ro+_bo+_go)
    _building = (gsi<0.15)
    building = (_building)&(~sky)&(~vegitation)
    # building = (_building)

    if (np.count_nonzero(sky)>np.count_nonzero(~sky)):
      bs[idx],gs[idx],rs[idx] = (bo[idx]*al+250*(1-al)),(go[idx]),(ro[idx]*al+100*(1-al))
      _sky[idx],mask[idx] = 0,0
    if (np.count_nonzero(vegitation)>np.count_nonzero(~vegitation)):
      bv[idx],gv[idx],rv[idx] = (bo[idx]*al+100*(1-al)),(go[idx]*al+200*(1-al)),(ro[idx])
      _veg[idx],mask[idx] = 0,0
    # if (np.count_nonzero(rubble)>np.count_nonzero(~rubble)):
    if (rubble):
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
