# ベクトル量子化で航跡のコード文書集合を作成する
import numpy as np
import scipy.cluster
import pickle
import datetime

# データファイルを読み込む
def readsailaisdata():
    with open("./aisdata/sailsbysailid.pickle", mode="rb") as fin:
        sailsbysailid = pickle.load(fin)
    return sailsbysailid

# コードブックを作成
def makecodebook(sailsbysailid, codebooklen):
    # すべてのセイルデータを結合
    firstflag = 0
    for sailid, saildata in sorted(sailsbysailid.items(), key=lambda x:x[0]):
        if firstflag == 0:
            saildataall = saildata
            firstflag=1
        else:
            saildataall = np.vstack((saildataall, saildata))
    print("コードブックを作成......")
    codebook, distortion = scipy.cluster.vq.kmeans(saildataall, codebooklen, iter=20, thresh=1e-05)
    print("codebook =\n", codebook)
    print("distortion =", distortion)
    # コードブックを保存
    codebookfile = "./aisdata/codebook"+str(codebooklen)+".pickle"
    with open(codebookfile, mode="wb") as fout:
        pickle.dump(codebook, fout)
    print("コードブックを"+codebookfile+"に保存しました。")
    return codebook

# コードの文書集合を作成 
def makedocuments(sailsbysailid, codebook, codebooklen):
    docsdata = []
    allcodelen = 0
    for sailid, saildata in sorted(sailsbysailid.items(), key=lambda x:x[0]):
        sailcode, dist = scipy.cluster.vq.vq(saildata, codebook)
        codedoc = [str(code) for code in sailcode]
        docsdata.append(codedoc)
        allcodelen += len(codedoc)
    print("文書に含まれるコードの総数 =",allcodelen)
    # コードの文書集合を保存
    docsfile = "./aisdata/docsdata"+str(codebooklen)+".pickle"
    with open(docsfile, mode="wb") as fout:
        pickle.dump(docsdata, fout)
    print("コードの文章集合を"+docsfile+"に保存しました。")
    return docsdata

# メイン関数
def main():
    # コードブックの長さを設定
    nn = 16
    codebooklen = nn*nn*nn*nn
    print("nn=", nn)
    print("コードブックの長さ =", codebooklen)
    sailsbysailid = readsailaisdata()
    # コードブックを作成
    codebook = makecodebook(sailsbysailid, codebooklen)
    print("実際のコードの数 =", len(codebook))
    # コードの文書集合を作成
    docsdata = makedocuments(sailsbysailid, codebook, codebooklen)
    print("コード文書の総数 =",len(docsdata))
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")