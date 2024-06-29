# 画像をセルに分割して、ベクトル量子化を行い、コードで文書集合を作成する
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import glob 
import scipy.cluster
import pickle
import datetime

# Calthch101の画像を保存するルートフォルダ
imagerootdir = "./caltech-101/101_ObjectCategories/"
# 利用する画像のカテゴリ（サブフォルダ）を指定
imagecategories = ["accordion", "butterfly", "cup"]
# ベクトル量子化のユニークなコード数
num_codes = 512
# セルのサイズ
cellsize = 8

#画像を読み込む
def read_images(filenames):
    # 画像ファイルの読み込み
    images = []
    for filename in filenames:
        image = plt.imread(filename)
        imshape = image.shape
        if len(imshape) == 3: # 3次元（カラー画像）のみ対象
            images.append(image)
        else:
            print("取り入れ対象外の画像：", filename)
    print("取り入れた画像の数：", len(images))
    return images

# 画像をセルに分割して、それぞれをベクトルに変換
def image2cellvec(images):
    allimgcellvecs = [] 
    for image in images:
        num_row, num_col, num_lay = image.shape
        imgcellvecs=[]
        for i in range(0, num_row-1, cellsize):
            for j in range(0, num_col-1, cellsize):
                cell = image[i:i+cellsize, j:j+cellsize, :]
                # 3次元配列を１次元に
                cellvec = cell.reshape([1, -1])[0]
                if len(cellvec) == cellsize*cellsize*3:
                    imgcellvecs.append(cellvec)
        allimgcellvecs.append(imgcellvecs)
    return allimgcellvecs

# ベクトル量子化してから、コードの文書集合を作成する
def makedocuments(allimgcellvecs):
    vecsall = np.vstack(allimgcellvecs)
    npvecsall = np.array(vecsall, dtype='float64')
    print("コードブックを作成 ...")
    codebook, distortion = scipy.cluster.vq.kmeans(npvecsall, num_codes, iter=30, thresh=1e-06)
    filename = "image-BovW-codebook.pickle"
    with open(filename, "wb") as f:
        pickle.dump(codebook, f)
    print("コードブックを保存しました。")
    # ベクトル量子化
    # 各データをセントロイドに分類する
    codeall = []
    for imgcellvecs in allimgcellvecs:
        codes, dists = scipy.cluster.vq.vq(imgcellvecs, codebook)
        codeall.append(codes)
    # コードの文書集合を作成
    documents = []
    wordcnt = 0
    for codes in codeall:
        words = []
        for code in codes:
            words.append(str(code))
        wordcnt += len(words)
        documents.append(words)
    print("文書（画像）の数 =", len(documents))
    print("単語（コード）の数 =", wordcnt)
    filename = "image-BovW-documents.pickle"
    with open(filename, "wb") as f:
        pickle.dump(documents, f)
    return

# メイン関数
def main():
    #imagedirにある全ファイル名を取得する
    filenames = [x for category in imagecategories for x in glob.glob(join(imagerootdir+category+"/", '*.jpg'))]
    images = read_images(filenames)
    allimgcellvecs = image2cellvec(images)
    makedocuments(allimgcellvecs)
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")