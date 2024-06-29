# トピックを航跡に復元して、地図に描く
import numpy as np
import pickle
from gensim import models
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
import japanize_matplotlib
import datetime

# ブレスト湾外の海域
west = -6.00 # 経度（西（左））
east = -3.00 # 経度（東（右））
south = 46.75 # 緯度（南（下））　
north = 48.75 # 緯度（北（上））
lonscale = 0.50
latscale = 0.25

# トピックデータを読み込む
def readtopicsdata(model, corpus, num_topwords):
    toptopics = model.top_topics(corpus, topn=num_topwords)
    topicsdata = {}
    for topicno, topic in enumerate(toptopics):
        topdata = {}
        for topprob, topcode in topic[0]:
            topdata[topcode] = float(topprob)
        topicsdata[topicno] = topdata	
    print( "トピック数 =", len(topicsdata) )
    return topicsdata

# コードブックを引いて、トピックのコードをセントロイド（経度、緯度、船首方位、速力）に戻す
def maketopictrajectory(topicdatadic, vqcodebook):
    topicstrajdic = {}
    for topicno, topdata in topicdatadic.items():
        lons = []
        lats = []
        headings = []
        speeds = []
        for topcode, topprob in sorted(topdata.items(), key=lambda x: x[1], reverse=True):
            centoroid = vqcodebook[int(topcode)]
            lons.append(float(centoroid[0]))
            lats.append(float(centoroid[1]))
            headings.append(float(centoroid[2]))
            speeds.append(float(centoroid[3]))
        topicstrajdic[topicno]=[lons, lats, headings, speeds]
    return topicstrajdic

# トピック番号別に軌跡を描く
def drawtrajectory(topicno, topictraj, num_topics):
    lons = np.array(topictraj[0])
    lats = np.array(topictraj[1])
    headings = np.array(topictraj[2])
    speeds = np.array(topictraj[3])
    r0 = 0.06
    sita0 = (90.0-headings)*np.pi/180.0
    u0 = r0*np.cos(sita0)
    v0 = r0*np.sin(sita0)
    # 軌跡を地図に表示
    plt.figure(figsize=(6, 6))
    m = Basemap(projection="merc", llcrnrlat=south, llcrnrlon=west, urcrnrlat=north, urcrnrlon=east, resolution="f")
    # 陸と海岸線と海
    m.drawcoastlines(color="k", linewidth=0.3)
    m.fillcontinents(color="y",lake_color="b")
    m.drawmapboundary()
    # 経線と緯線
    m.drawmeridians(np.arange(west, east, lonscale),labels=[0,0,0,1], linewidth=0.3, fontsize=9)
    m.drawparallels(np.arange(south, north, latscale),labels=[1,0,0,0], linewidth=0.3, fontsize=9)
    U, V, X, Y = m.rotate_vector(u0, v0, lons, lats, returnxy=True)
    # 矢印（ベクトル）
    cs1 = m.quiver(X, Y, U, V, speeds, norm=Normalize(vmin=0, vmax=40), edgecolor="k",  linewidth=.1, scale_units="inches", scale=1)
    # カラーバー
    bounds = np.array([0, 5, 10, 20, 30, 40, 50])
    cbar = m.colorbar(cs1,location="right", pad="10%", ticks=bounds)
    cbar.set_label("速力(ノット)")
    plt.title("トピック番号 ="+str(topicno)+"/"+str(num_topics), fontsize=16)
    plt.savefig("./aistopics/trajectory-topics"+str(num_topics)+"-no"+str(topicno)+".png", dpi=720)
    plt.close()
    return 

#　メイン関数
def main():
    # パラメータ
    num_topics = 20
    num_topwords = 200
    nn = 16
    codebooklen = nn*nn*nn*nn
    codebookfile = "./aisdata/codebook"+str(codebooklen)+".pickle"
    with open(codebookfile, mode="rb") as fin:
        codebook=pickle.load(fin)
    # 保存済みのデータを読み込む
    corpfile = "./aistopics/AIS-documents-corpus.pickle"
    with open(corpfile, "rb") as f:
        corpus = pickle.load(f)
    # 訓練済みのモデルをロードする
    modelfile = "./aistopics/topics"+str(num_topics)+".model"
    model = models.ldamodel.LdaModel.load(modelfile)
    # トピック別の上位コードを取得
    print("トピック別の上位コードを取得 ...")
    topicsdatadic = readtopicsdata(model, corpus, num_topwords)
    # トピック別の上位コードをセントロイド（経度、緯度、船首方位、速力）に復元
    print("トピック別の上位コードを経度、緯度、船首方位、速力に復元 ...")
    topicstrajdic = maketopictrajectory(topicsdatadic, codebook)
    # トピックの航跡を地図に描く
    print("トピックの航跡を地図に描く ...")
    for topicno, topictraj in topicstrajdic.items():
        print( "トピック番号 =",topicno )
        drawtrajectory(topicno, topictraj, num_topics)
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")