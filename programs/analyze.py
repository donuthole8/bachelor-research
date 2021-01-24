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
#   - ヒストグラム均一化を先にやった方がいいかも
# methods.meanshift(12,3,200)
# methods.meanshift(img,12,7,500)

# PyMeanShift実行時間短縮
# methods.shortcut1()
methods.shortcut2()
# methods.shortcut3()

# ヒストグラム均一化
#   - Lab変換して処理した方が好ましい可能性がある
#   - 明度以外の要素も均一化した方が好ましい可能性がある
methods.equalization()


# クロージング処理



# クロージング処理！！！！！
# 使うべし～～～、カラーだとむずいかもおおお
# methods.closing
# 精度評価で地図と位置合わせできたら国土地理院の斜面崩壊判読図使うのもよし



# 量子化・減色処理・ノイズ除去
#   - 領域分割後に類似領域に異色画素が散在してるため
# img = methods.quantization(img)
# img = cv2.medianBlur(img,5)
# img = cv2.bilateralFilter(img,9,75,75)

# 類似色統合
# imgに入れるんじゃなくてカラーラベリングするときだけ使う方がよい？？？
methods.clustering()

# カラーラベリング
# methods.labeling()

# カラーラベリング実行時間短縮
methods.shortcut_2()

# テクスチャ解析
#   - img か org のどちらに適用するか関数内で選択
methods.texture()

# エッジ抽出
#   - img か org のどちらに適用するか関数内で選択
methods.edge()

# 災害領域検出
#   - 斜面崩壊・瓦礫でピクセル穴がある　
#   - 浸水検出結果で緑色のノイズがある
lnd,fld = methods.detection()

# 不要領域検出
mask = methods.rejection()

# 最終出力
_lnd,_fld = methods.integration(mask,lnd,fld)

# 精度評価
# methods.evaluation(_lnd,_fld)
