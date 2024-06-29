# ユニグラムモデル、単語の分布を推定
import numpy as np
import pymc as pm
from matplotlib import pyplot as plt
import arviz as az
import pickle
import datetime

# メイン関数
def main():
    # 文書データを読み込む
    with open("tpm-ch5ex1-documents.pickle", "rb") as pklin:
        documents = pickle.load(pklin)
    print("文書集合：")
    print(documents)
    allwords = " ".join(documents)
    allwords = allwords.split()
    print(allwords)
    # 単語帳を読み込む
    with open("tpm-ch5ex1-vocabulary.pickle", "rb") as pklin:
        vocabulary = pickle.load(pklin)
    print("単語帳：")
    print(vocabulary)
    V = len(vocabulary)
    #単語を単語番号に変換
    docsdata = []
    for doc in documents:
        words = doc.split()
        for word in words:
            for id, value in enumerate(vocabulary):
                if value == word:
                    break
            docsdata.append(id)
    # リストをnumpy配列にする
    observed_data = np.array(docsdata)
    print("observed_data =")
    print(observed_data)
    # ベイズ分析
    with pm.Model() as unigram_model:
        phi = pm.Dirichlet("phi", a=np.ones(V))
        sampledata = pm.Categorical("sampledata", p=phi, observed=observed_data)
    with unigram_model:
        idata = pm.sample(draws=5000, tune=1000, chains=4, progressbar=True)
        summary_idata = az.summary(idata, var_names=["phi"], kind="stats")
        print("サマリー:")
        print(summary_idata)
        az.plot_posterior(idata, hdi_prob=0.97)
        figfile = "tpm-ch5ex3-fig1.png"
        plt.savefig(figfile)
        print("グラフを"+figfile+"に保存しました。")
        print("パラメータ（単語の出現確率）の推定結果：")
        posterior = az.extract(idata)
        phi_posterior = posterior['phi'].values
        mm, nn = phi_posterior.shape
        words_prob = phi_posterior[:,nn-1]
        for i in range(V):
            print("単語=", vocabulary[i], "\t\t 出現確率=", words_prob[i].round(3))
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")
    