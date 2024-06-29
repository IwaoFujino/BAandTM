# トピックモデルの処理結果（トピック別の上位セルのコードと画像）を表示する
from gensim import  models
import pickle
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import datetime

# パラメータ
num_topics = 20

# メイン関数
def main():
    # 保存済みのデータを読み込む
    print("辞書とコーパスをロード...")
    dictfile = "image-BoVW-dictionary.pickle"
    with open(dictfile, "rb") as f:
        dictionary = pickle.load(f)
    corpfile="image-BoVW-corpus.pickle"
    with open(corpfile, "rb") as f:
        corpus = pickle.load(f)
    # 訓練済みのモデルをロードする
    print("モデルをロード...")
    modelfile = "image-BoVW-lda"+str(num_topics)+".model"
    model = models.ldamodel.LdaModel.load(modelfile)
    # 各トピックの上位単語を取得する
    codebookfile="image-BoVW-codebook.pickle"
    with open(codebookfile, "rb") as f:
        codebook = pickle.load(f)
    print("トピック別の上位単語：")
    # 各トピックの上位単語を表示する
    toptopics = model.top_topics(corpus=corpus, dictionary=dictionary, topn=10)
    print("コヒーレンスの降順でソートしたトピック：")
    alltopcodes=[]
    alltopcells=[]
    for topicid, topic in enumerate(toptopics):
        print("-----", topicid, "番のトピック -----")
        print("コヒーレンス =", topic[1].round(6))
        topcells=[]
        topcodes=[]
        for tp in topic[0]:
            code = tp[1]
            prob = tp[0]
            print(code, "\t", prob)
            centroid = np.uint8(codebook[int(code)])
            cellimg = centroid.reshape(8, 8, 3)
            topcodes.append(code)
            topcells.append(cellimg)
        alltopcodes.append(topcodes)
        alltopcells.append(topcells)
    # トップコードの画像セルを表示 (２ページ分)
    for page in range(2):
        plt.figure(figsize=(6, 8))
        plt.subplots_adjust(wspace=0.4, hspace=1.0)
        for rowid in range(10):
            topicid = page*10 + rowid
            for i in range(10):
                plt.subplot(10, 10, rowid*10+i+1)  
                plt.imshow(alltopcells[topicid][i])
                plt.axis('off')
                plt.title(alltopcodes[topicid][i])
        plt.suptitle("トピック別の上位10 番までのコードとセルイメージ( ページ"+ str(page) + ")", y=0.95, fontsize=14)
        plt.savefig("tpm-ch11ex6-fig"+str(page)+".png", dpi=360)
        plt.close()
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")