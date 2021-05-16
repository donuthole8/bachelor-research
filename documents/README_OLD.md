<div style="text-align: center;"><h1>卒業研究</h1></div>

## 研究テーマ
ドローン空撮動画像を用いた山間部における豪雨時の斜面崩壊及び浸水の検出


## 概要

豪雨災害後のドローン空撮映像を用いて斜面崩壊領域と浸水領域を検出する


## 手順
![スライド1](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\スライド1.JPG)


## 使用データ

- [国土地理院](https://www.gsi.go.jp/BOUSAI/H29hukuoka_ooita-heavyrain.html#5)からの無料配布データ使用


- ドローン映像（原映像）

<video src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\土砂災害7_Trim.mp4" width="500"></video>


- ドローン画像（フレーム分割処理後）

![madslide10](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\madslide10.jpg)

![madslide12](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\madslide12.jpg)


- 使用するドローン映像は豪雨災害のため、斜面崩壊地と植生以外にも倒壊家屋や道路、**水害**の映り込みも多い

- 映り込み領域
  - 植生領域
    - 森林、畑、植物等
    - 枯れ草、枯れ木等
      - 写ってないはず
      - 写っていたとしたら土砂との判別が難しそう
  - 土砂領域
    - 土砂、岩石、裸地畑等
    - 斜面崩壊領域
    - 浸水領域
  - 空領域
    - 空、雲
  - 人工物領域
    - 家屋、道路、瓦礫、車、コンクリート、電柱、鉄塔、標識、電線等
  - その他
    - 木材・倒木領域
      - 人工物領域・土砂領域のどちらに分類すべきか未定
    - 人物、水、影

- 最終的に検出したい領域

  - 斜面崩壊領域
  - 浸水領域


- 三次元モデル

<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\オルソ.png" width="500">


- オルソ補正画像

<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\olso.jpg" width="500">


## 手法
### フレーム画像切り取り
- [Free Video to JPG Converter](https://www.gigafree.net/media/conv/freevideotojpgconverter.html) を使用
- ドローン空撮映像からフレーム画像を切り取る
- 空撮画像用フレーム画像
  - 1枚
- 三次元モデル生成用フレーム画像
  - 100枚程度
  - もっと少なくても生成できるかも


![Free Video to JPG Converter](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\Free Video to JPG Converter.jpg)


### 前処理

- ヒストグラム平坦化
  - openCVによって実装

```python
# HSV変換
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# CLAHEパラメータの設定
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
# V値のヒストグラム平坦化
hsv[:,:,2] = clahe.apply(hsv[:,:,2])
```


![madslide20](C:\Users\cs17090\Saji-Lab\researchSaji\images\madslide20.jpg)

<div style="text-align: center;"><p>ヒストグラム平坦化前</p></div>


![histo](C:\Users\cs17090\Saji-Lab\researchSaji\results\histo.jpg)

<div style="text-align: center;"><p>ヒストグラム平坦化後</p></div>


- 画像のノイズ除去

  - [画像のノイズ除去](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html)を参考にしたノンローカルミーンフィルタ

  - メディアンフィルタ、バイラテラルフィルタ等も試したがこれが一番良さそう

``` python:main.py
img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
```




![madslide6](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\madslide6.jpg)

<div style="text-align: center;"><p>ノイズ除去処理前</p></div>




![noise](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\noise1.jpg)


<div style="text-align: center;"><p>ノイズ除去処理後</p></div>



### 領域分割
- mean-shift法にて領域分割
- [Mean Shift Segmentation の OpenCV 実装](http://visitlab.jp/pdf/MeanShiftSegmentation.pdf)を参考にmean-shift法を使用
``` python:main.py
img = cv2.pyrMeanShiftFiltering(img, 32, 32)
```
- 同様の色相を持つ領域を一つの領域としてまとめる





![noise2](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\noise2.jpg)


<div style="text-align: center;"><p>mean-shift法処理前</p></div>



![meanshift1](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\meanshift1.jpg)


<div style="text-align: center;"><p>mean-shift法処理後</p></div>



- PyMeanShift
  - Windows環境では少し面倒
  - パソコンの調子が悪くVirtualBoxが重い
  - 研究室のデスクトップが使えるようになってから実装したい



<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\cat.jpg" width=500>

<div style="text-align: center;"><p>PyMeanShift処理前</p></div>



<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\cat_meanshift.png" width=500>

<div style="text-align: center;"><p>PyMeanShift前</p></div>





### 領域判別

- この処理時点ではL\*a\*b\*表色系を用いて植生・災害候補（後に斜面崩壊・浸水・土砂・人工物に分類）・空に分類
  - 植生　　（緑）：a値低い画素を植生領域とする
  - 土砂候補（赤）：a値高い画素を災害候補領域とする
  - 空　　　（青）：b値低い画素を空領域とする



![Lab](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\Lab.png)

<div style="text-align: center;"><p>L*a*b*表色系概念図</p></div>



- 他の指標も試したがL\*a\*b\*表色系による分類が最も高精度
  - 緑過剰指標
  - 緑赤植生指標
  - RGBの最大値
  - HSVの色相
- テクスチャ特徴・輝度値等も使用して分類精度向上
  - 只今実装中...



![pseudo](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\pseudo.jpg)

<div style="text-align: center;"><p>異質度画像</p></div>



- 異質度低：青　異質度高：赤
- 失敗してる気がする



### 空・植生領域除去

- 領域分類にて検出した空・植生領域を最終結果から除去するためマスク画像を作成



![maskBG](C:\Users\cs17090\Saji-Lab\researchSaji\results\meanshift.jpg)

<div style="text-align: center;"><p>マスク画像作成前画像</p></div>



<img src="C:\Users\cs17090\Saji-Lab\researchSaji\results\maskBG.jpg" width=500 style="border: 1px black solid;">

<div style="text-align: center;"><p>マスク画像</p></div>



### DEMデータとフレーム画像との位置合わせ

- 中止



### 三次元モデル生成

- ドローン映像のフレーム画像数十枚から三次元モデル生成
- [Metashape](https://www.too.com/product/software/cgcad/metashape/)（旧 PhotoScan） を使用
  - 研究室の計算機サーバにある
  - 三次元復元・DEM生成・オルソ画像生成等が可能になる高性能三次元復元ソフト



![Metashape](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\Metashape.jpg)





<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\フレーム画像.png" width=500>

<div style="text-align: center;"><p>フレーム画像</p></div>



<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\三次元モデル.png" width=500>

<div style="text-align: center;"><p>三次元モデル</p></div>



### オルソ補正

- 未実装



### 人工物領域検出

- 災害候補領域中から人工物領域を除去する

  - 植生領域・空領域は既に最終結果に関係ない

  - 人工物は赤みが強いことが多いため災害候補領域と誤検出されてしまう

  - 色相・色味による判別が難しいので特徴量や指標を用いたいところ

    

- 人工物領域除去手法候補

  - 方法１：テクスチャ解析
    
    - 空撮画像を用いた土砂領域検出（中山さん）の手法
    
    - テクスチャ特徴である異質度を用いる
    
      - 画素が不均一であるほど異質度は高い値となる
      - 人工物領域は異質度が低い
    
      

<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\空撮画像.png" width="500">

<div style="text-align: center;"><p>入力画像</p></div>



<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\異質度.png" width="500">

<div style="text-align: center;"><p>異質度計算結果（本来はグレースケール画像）</p></div>
<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\疑似カラー.png" width="500">
<div style="text-align: center;"><p>疑似カラー</p></div>



<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\異質度マスク画像.png" width="500">

<div style="text-align: center;"><p>異質度マスク画像</p></div>



- 異質度が低い領域をマスク画像として土砂候補領域から除去

  

  - 方法２：エッジによる建物領域・瓦礫領域抽出
      - エッジ特徴量を用いた地震後上空画像による建物被害状況解析（遠藤さん）の手法
      
      - 建物が倒壊した後の瓦礫領域は大量の短いエッジが存在する
      
      - 建物が倒壊せず形状を維持している領域は連結した長いエッジが存在する
      
      - [Cannyのエッジ抽出手法](https://qiita.com/Takarasawa_/items/1556bf8e0513dca34a19)を用いて細線化
      
      - エッジの持つ特徴量毎に「赤：非倒壊」「緑：半壊」「青：倒壊」に分類
      
        

<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\建物入力画像.png" width="500">

<div style="text-align: center;"><p>入力画像</p></div>



<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\建物エッジ画像.png" width="500">

<div style="text-align: center;"><p>エッジ抽出結果</p></div>



<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\建物検出結果.png" width="500">

<div style="text-align: center;"><p>結果画像</p></div>



### オルソ画像との位置合わせ

- 未実装



### 人工物領域除去

- 未実装




### 斜面崩壊・浸水検出
- 斜面崩壊領域・浸水領域・その他に判別し斜面崩壊及び浸水領域を最終出力とする

  

- 斜面崩壊領域の検出手法候補

  - 方法１：土砂候補領域から浸水領域や人工物領域を除去して残った領域を斜面崩壊領域とする
    - 浸水領域判別が高精度でできたら使えそう
  - 方法２：何らか特徴量を用いる
    - 異質度・均一度・分散・エッジ等
  - 方法３：災害候補領域中から輝度値を用いて判別
    - 誤検出も多い



- 浸水領域の検出手法候補（どれかを使用或いは併用）

    - 方法１：エッジ抽出・テクスチャ解析にて浸水領域判別
    
      - ヘリコプター映像を利用した浸水道路領域の抽出（小笠原さん）、浸水時における道路領域解析（雨宮さん）の手法
    
      - エッジ抽出
        - 浸水領域ではエッジが少なくなる（道路領域のみかも）
    
      - テクスチャ解析
        - ２次統計量によるテクスチャ指標によって画素の方向性等の性質を表す特徴量が求まる
        - 異質度を利用
      - 明るい画素と暗い画素が隣接する箇所で値が大きくなる
        - 浸水領域では値が小さくなる
    
      - 「エッジ抽出率：10%以下」かつ「低異質度率が50%以上」の領域を浸水領域として判別
    
      

<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\浸水入力画像.png" width="500">

<div style="text-align: center;"><p>入力画像</p></div>




<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\エッジ抽出.png" width="500">
<div style="text-align: center;"><p>エッジ抽出処理</p></div>




<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\テクスチャ解析.png" width="500">
<div style="text-align: center;"><p>テクスチャ解析処理</p></div>




<img src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\浸水領域判別.png" width="500">
<div style="text-align: center;"><p>浸水領域判別処理</p></div>

​	

- - 都市部のような道路上でしか検出できない可能性あり
- - 画像を見ると今回の泥水のような浸水で無く透明な水の浸水なので上手くいかない可能性あり


​      

  - 方法２：土砂候補領域中の輝度が高い画素を浸水領域とする

    - 浸水領域は水分を含む分、太陽光を反射して輝度が明るいと感じた

    - 天候や影等の撮影条件に左右されると思われる

    - 画像毎に誤検出がひどい

        

        <div style="text-align: center;"><p>緑：植生領域　赤：斜面崩壊領域　黄色：浸水領域　青：空領域</p></div>

    

    ![madslide12](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\madslide12.jpg)

    ![result1](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\result.jpg)

      <div style="text-align: center;"><p>比較的上手く浸水領域を判別できた画像</p></div>

    
    
	![madslide2](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\madslide2.jpg)
	
	![result2](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\result22.jpg)

  <div style="text-align: center;"><p>失敗画像</p></div>



  - 方法４：フレーム画像と地図・道路データを位置合わせして道路や家屋が無くなっている箇所を浸水領域とする

      - 手動で位置合わせ

          - できるかどうか及びやり方は不明

      - 地図・道路データにおいて道路や家屋を判別する必要がある






## 実装結果

- 使用ドローン映像（実際は２分程）

	<video src="C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\土砂災害7_Trim.mp4" width="500"></video>

  <div style="text-align: center;"><p>ドローン映像</p></div>



- フレーム画像切り出し

  ![madslide12](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\madslide12.jpg)

  <div style="text-align: center;"><p>フレーム画像</p></div>



- 前処理

  ![noiseImg](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\noiseImg.jpg)

<div style="text-align: center;"><p>ノイズ除去画像</p></div>



- 領域分割

  ![meanshiftImg](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\meanshiftImg.jpg)

<div style="text-align: center;"><p>mean-shift法処理後画像</p></div>



- 領域判別

  - 緑：植生領域

  - 赤：土砂候補領域（後の斜面崩壊領域・浸水領域・土砂領域・人工物領域）

  - 青：空領域

    

  ![segmentation](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\segmentation.jpg)

<div style="text-align: center;"><p>領域判別画像</p></div>



- 空・植生領域除去

<img src="C:\Users\cs17090\Saji-Lab\researchSaji\results\maskBG.jpg" width=500 style="border: 1px black solid;">

<div style="text-align: center;"><p>マスク画像</p></div>



- 人工物領域除去

  - 未完成
  
  - マスク画像とかにしたい
  
    
  
  ![edgeImg](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\edgeImg.jpg)

<div style="text-align: center;"><p>Canny法によるエッジ抽出画像</p></div>



- 斜面崩壊・浸水検出

  - 緑：植生領域
  - 赤：斜面崩壊領域
  - 黄：浸水領域
  - 青：空領域
  
  
  
  ![result](C:\Users\cs17090\Documents\Saji-lab\卒論_室永\ゼミ資料\images\result.jpg)

<div style="text-align: center;"><p>斜面崩壊領域・浸水領域検出結果</p></div>