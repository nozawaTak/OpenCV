import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt



def cvtBGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def cvtBGR2GRAY(img):
    GRAY_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.merge((GRAY_img, GRAY_img, GRAY_img), img)

def print_img(img):
    plt.imshow(img)
    plt.show()

def threshold(img):
    _, thresh = cv2.threshold(cvtBGR2GRAY(img), 67, 255, cv2.THRESH_TOZERO)
    return thresh

def morphology(img):
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel=(3, 3))

def test_movie():
    """
        動画を読み込んで表示する関数。
        まずVideoCapture()で動画オブジェクトを作成。
        そこからread()で１フレームずつ読み込むことで静止画と同じ扱いが可能になる。
        （動画は高速に連続する大量の静止画と考えれば理解を掴みやすい。）
        read()はフレームが読み込めればTrue、読み込めなければFalseを返すので、
        変数retがFalseを持った時が、動画を全部読み込み終わった時だと判別できる。
        waitKey()は引数ミリセカンド実行を停止するので、調整することで動画の再生スピードを調整できる。
        25くらいが普通の再生スピードになるらしい。
    """
    cap = cv2.VideoCapture('Vtest.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        if cv2.waitKey(50) and ret is not True:
            break
        cv2.imshow("frame", frame)

def moving_object_extraction():
    """
        動画中の動いている物体の輪郭を抽出する関数。
        グレースケールにした各フレームとその一個前のフレームからフレーム列の移動平均値を算出する。
        accumulateWeighted()は
    """
    cap = cv2.VideoCapture('Vtest.avi')
    avg = None
    while cap.isOpened() :
        ret, frame = cap.read()

        if cv2.waitKey(50) and ret != True:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if avg is None:
            avg = gray.copy().astype("float")
            continue
        cv2.accumulateWeighted(gray, avg, 0.9)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
        image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        cv2.imshow('frame', img)

def median_filter():
    """
        メディアンフィルタリング。平滑化の一種。
        cv2.medianBlur()を使う。Blurはぼかしの意味。
        各画素を周辺画素の中央値に置き換える。
        第２引数はカーネルサイズと呼ばれ、要は周辺画素の範囲。３なら３×３の９画素中の
        中央値を求める。注意としてカーネルサイズは奇数じゃないとダメ。偶数だと
        周辺範囲を定められない。確かに。
    """
    cap = cv2.VideoCapture('Vtest.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        if cv2.waitKey(50) and ret is not True:
            break
        med_frame = cv2.medianBlur(frame, 5)
        cv2.imshow("frame", med_frame)

def moving_averag_filter():
    """
        移動平均フィルタ。平滑化の基本的立ち位置だけどその性能は微妙。
        各画素を周辺画素との平均値に置き換える。
        第２引数は平均化する周辺画素の範囲で、２次元タプルで渡す。
        主観だけど、(x,y)のうちxを大きくすると横にぶれた感じ、yを大きくすると
        縦にぶれた感じになる。
    """
    cap = cv2.VideoCapture('Vtest.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        if cv2.waitKey(50) and ret is not True:
            break
        filter_size = (5, 5)
        maf_frame = cv2.blur(frame, filter_size)
        cv2.imshow("frame", maf_frame)

def gaussian_filter():
    """
        ガウシアンフィルタ。
        各画素を周辺画素との重み付け平均値で置き換える。
        重みは中心からの距離に応じて小さくなっていく。近いほど重みを大きく、
        遠いほど重みを小さくして掛け合わせる方式。
        重みはガウス分布に従うのでガウシアン。
        sigmaXはガウス分布の標準偏差の値。
    """
    cap = cv2.VideoCapture('Vtest.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        if cv2.waitKey(50) and ret is not True:
            break
        gaussian_frame = cv2.GaussianBlur(frame, ksize = (3, 3), sigmaX = 1.3)
        cv2.imshow("frame", gaussian_frame)


def moving_object_extraction_with_gaussian():
    """
        動画中の動いている物体の輪郭を抽出する関数をガウシアンフィルタを用いて平滑化してみた。
        結果、本来検出して欲しくない静止した物体を気持ち検出しにくくなったような。
        一方、移動物体に２重の輪郭線が描かれてしまうようになった。（パラメータ調整で直るかも？）
    """
    cap = cv2.VideoCapture('Vtest.avi')
    avg = None
    while cap.isOpened() :
        ret, frame = cap.read()

        if cv2.waitKey(50) and ret != True:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_gray = cv2.GaussianBlur(gray, ksize = (7,7), sigmaX = 1.3)
        if avg is None:
            avg = gaussian_gray.copy().astype("float")
            continue
        cv2.accumulateWeighted(gaussian_gray, avg, 0.8)
        frameDelta = cv2.absdiff(gaussian_gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
        image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        cv2.imshow('frame', img)

"""
    次回モルフォロジー（オープニングとクロージング）を試してみましょう
"""

if __name__ == "__main__":
    #img = cv2.imread('/Users/TAK/Desktop/test.jpg')
    #print_img(cvtBGR2RGB(img))
    #print_img(morphology(cvtBGR2RGB(img)))
    #print_img(cvtBGR2RGB(img))
    #print_img(cvtBGR2GRAY(img))
    #print_img(threshold(img))
    #test_movie()
    #moving_object_extraction()
    #median_filter()
    #moving_averag_filter()
    #gaussian_filter()
    moving_object_extraction_with_gaussian()
