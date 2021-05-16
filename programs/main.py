import cv2
import prepro
import analysis
import detection
import postpro


# 色んな画像に対して同時に処理するプログラム
# 或いは

# 入力画像読込
#   - png形式画像を用いるのが好ましい
#   - 本来の画像サイズ(1800x1080 pix)での処理が好ましい
# img = cv2.imread('images/madslide11.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('images/madslide12.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread('images/kuramaeya.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread('images/madslide20_.jpg', cv2.IMREAD_COLOR)

# オリジナル画像保存
# import pdb; pdb.set_trace()
org = img.copy()
cv2.imwrite('results/original.png',org)

# 画像ファイル
#   - 画像データを処理プログラムに送る
# prepro.image(org,img)

# PyMeanShift
#   - Lab変換後の画像に適用
#   - 第１引数：探索範囲、第２引数：探索色相、第３引数：粗さ
#   - ヒストグラム均一化を先にやった方がいいかも

# kihonnkore?
# img = prepro.meanshift(img,12,3,200)


# methods.meanshift(12,7,200)

# methods.meanshift(5,3,1000)
# methods.meanshift(15,5,500)
# methods.meanshift(12,7,700)

# PyMeanShift実行時間短縮
img = prepro.shortcut1()

# ヒストグラム均一化
#   - Lab変換して処理した方が好ましい可能性がある
#   - 明度以外の要素も均一化した方が好ましい可能性がある
img = prepro.equalization(img)




# ヒストグラム均一化と



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

# 閾値買えよおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおお
img = prepro.clustering(img)


# あと瓦礫は被災領域にしちまえよおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおおお
# 地図上で通れない箇所だし、下で埋もれてる人もいるかもだし！！！！！


# カラーラベリング
# 絶対読んで！！！！
# 呼んでｗｗｗｗｗｗｗ
# 減色処理してからラベリングしろ！！！！！！！

# 着目色（64色に減色）ごとに二値化（着目色とそれ以外、白黒ラベリングする）、そいで次の色に写って、、、、繰り返し
# 白黒ラベリング


# モルフォロジー処理カラーでできる？


qua = prepro.quantization(img)

# analysis.labeling(img,qua)

# カラーラベリング実行時間短縮
analysis.shortcut2()

# テクスチャ解析
#   - img か org のどちらに適用するか関数内で選択
# methods.texture(img)
# methods.texture(org)

# エッジ抽出
#   - img か org のどちらに適用するか関数内で選択
# analysis.edge(img)
analysis.edge(org)

# 災害領域検出
#   - 斜面崩壊・瓦礫でピクセル穴がある　
#   - 浸水検出結果で緑色のノイズがある
lnd,fld = detection.detection(org,img)

# 不要領域検出
# mask = methods.rejection()
mask,sky,veg,rbl,bld = detection.rejection(org,img)

# 最終出力
_lnd,_fld = postpro.integration(mask,lnd,fld,sky,veg,rbl,bld,org)

# 精度評価
#   - 領域単位で評価するのもあり
postpro.evaluation(_lnd,_fld)
