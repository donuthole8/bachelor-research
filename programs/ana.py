import cv2
import numpy as np
import matplotlib.pyplot as plt
import methods


# 入力画像読込
#   - png形式画像を用いるのが好ましい
#   - 本来の画像サイズ(1800x1080 pix)での処理が好ましい
# img = cv2.imread('images/madslide11.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('images/madslide12.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread('images/smallimage.jpg', cv2.IMREAD_COLOR)

# オリジナル画像保存
# import pdb; pdb.set_trace()
org = img.copy()
cv2.imwrite('results/original.png', org)

# 画像ファイル
#   - 画像データを処理プログラムに送る
methods.image(org,img)

# PyMeanShift
#   - Lab変換後の画像に適用
#   - 第１引数：探索範囲、第２引数：探索色相、第３引数：粗さ
methods.meanshift(12,3,200)

# ヒストグラム均一化
#   - Lab変換して処理した方が好ましい可能性がある
#   - 明度以外の要素も均一化した方が好ましい可能性がある
methods.equalization()

# 類似色統合
methods.clustering()

# カラーラベリング
methods.labeling()

# 災害領域検出
#   - 斜面崩壊・瓦礫でピクセル穴がある　
#   - 浸水検出結果で緑色のノイズがある
lnd,fld = methods.detection()