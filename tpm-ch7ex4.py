# トピックモデル
# 各文書のトピックを割合順に表示
# 各文書のトピックの割合を横棒グラフに
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import japanize_matplotlib
import pickle
import datetime

# 各トピックの上位単語を表示
def print_docs_topic(thetas):
    for docno, doctopics in enumerate(thetas):
        print("文書番号=", docno)
        doctopicsdf = pd.DataFrame(doctopics, index=["トピック0", "トピック1", "トピック2"], columns=["割合"])
        print(doctopicsdf.sort_values("割合", ascending=False))
    return

# 各トピックの棒グラフ
def barhr_docs_topic(thetas):
    docstopicsdf = pd.DataFrame(index=["トピック0", "トピック1", "トピック2"])
    for docno, doctopics in enumerate(thetas):
        docstopicsdf["文書"+str(docno)]=doctopics.values
    docstopicsdf = docstopicsdf.transpose()
    docstopicsdf.plot.barh(grid=True)
    figfile = "tpm-ch7ex4-fig1.png"
    plt.savefig(figfile)
    print("グラフを"+figfile+"に保存しました。")
    return

# メイン関数
def main():
    # 文書データを読み込む
    with open("tpm-ch7ex2-idata.pickle", "rb") as pklin:
        idata = pickle.load(pklin)
    theta_posterior = az.extract(idata, var_names=["theta"])
    kk, mm, nn = theta_posterior.shape
    # 文書別のトピックの割合を降順に表示
    print_docs_topic(theta_posterior[:,:,nn-1])
    # 文書別のトピックの割合の横棒グラフを作成
    barhr_docs_topic(theta_posterior[:,:,nn-1])
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")