# 混合ユニグラムモデル
# 各トピックの単語を出現確率順に表示
# 各トピックの単語を出現確率を横棒グラフに
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import japanize_matplotlib
import pickle
import datetime

# Pandas画面表示用：小数点以下４桁
pd.options.display.float_format = '{:.6f}'.format

# 各トピックの上位単語を表示
def print_words_prob(vocab, phis):
    for tpno, tpprob in enumerate(phis):
        print("トピック=", tpno)
        wordsprobdf = pd.DataFrame(tpprob, columns=["prob"], index=vocab)
        print(wordsprobdf.sort_values("prob", ascending=False))
    return

# 各トピックの棒グラフ
def barhr_words_prob(vocab, phis):
    tpprobdf = pd.DataFrame(index=vocab)
    for tpno, tpprob in enumerate(phis):
        tpprobdf["トピック"+str(tpno)]=tpprob.values
    tpprobdf.plot.barh(grid=True)
    figfile = "tpm-ch6ex3-fig1.png"
    plt.savefig(figfile)
    print("グラフを"+figfile+"に保存しました。")
    return

# メイン関数
def main():
    #単語リストを作成
    with open("tpm-ch6ex1-vocabulary.pickle", "rb") as pklin:
        vocabulary = pickle.load(pklin)
    # 文書データを読み込む
    with open("tpm-ch6ex2-idata.pickle", "rb") as pklin:
        idata = pickle.load(pklin)
    phi_posterior = az.extract(idata, var_names=["phi"])
    kk, mm, nn = phi_posterior.shape
    print_words_prob(vocabulary, phi_posterior[:,:,nn-1])
    barhr_words_prob(vocabulary, phi_posterior[:,:,nn-1])
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")