import cv2
import pymeanshift as pms
from PIL import Image


def meanshift(img,spatial_radius,range_radius,min_density):
  lab_img, _, _ = pms.segment(cv2.cvtColor(img, cv2.COLOR_BGR2Lab), spatial_radius, range_radius,min_density)
  img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)

  cv2.imwrite('results/meanshift.png',img)
  print('meanshift done')
  return img


def shortcut1():
  img = cv2.imread('results/meanshift.png', cv2.IMREAD_COLOR)
  # img = cv2.imread('results/meanshift500.png', cv2.IMREAD_COLOR)
  # img = cv2.imread('results/meanshift1920.png', cv2.IMREAD_COLOR)
  return img


# def image(_org,_img):
#   global org,img,h,w,c
#   global bo,go,ro,al

#   org,img = _org,_img
#   h,w,c = img.shape
#   bo,go,ro = cv2.split(org)
#   al = 0.45


def equalization(img):
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv_img)

  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
  new_v = clahe.apply(v)

  hsv_clahe = cv2.merge((h, s, new_v))
  new_rgb_img = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

  img = new_rgb_img
  cv2.imwrite('results/equalization.png',img)

  return img


def clustering(img):
  img = Image.fromarray(img)
  img_q = img.quantize(colors=128, method=0, dither=1)
  img_q.save('results/clustering.png')
  img = cv2.imread('results/clustering.png', cv2.IMREAD_COLOR)

  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  cv2.imwrite('results/clustering.png',img)

  return img


def quantization(img):
  img = img // 4 * 4
  # img = img // 8 * 8
  cv2.imwrite('results/quantization.png', img)

  return img
