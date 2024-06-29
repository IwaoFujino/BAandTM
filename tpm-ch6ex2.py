# 混合ユニグラムモデル
# ディリクレ分布を用いて、カテゴリ分布のパラメータを推定
# 推定結果をファイルに保存する
import numpy as np
import pymc as pm
import arviz as az
import pickle
import datetime

# メイン関数
def main():
    # 文書集合を読み込む
    with open("tpm-ch6ex1new-documents.pickle", "rb") as pklin:
        documents = pickle.load(pklin)
    #print("文書別の単語：")
    #print(documents)
    #単語リストを作成
    with open("tpm-ch6ex1new-vocabulary.pickle", "rb") as pklin:
        vocabulary = pickle.load(pklin)
    print("単語帳：")
    print(vocabulary)
    V = len(vocabulary)
    #単語を単語番号に変換
    docsdata = []
    D = len(documents)
    N = []
    for doc in documents:
        words = doc.split()
        N.append(len(words))
        for word in words:
            wordid = vocabulary.index(word)
            docsdata.append(wordid)
    # 文書データを表示
    #print("文書データ：")
    #print(docsdata)
    print("文書数D=",D)
    print("各文書の単語数N=",N)
    print("単語帳の単語数V=",V)
    # 単語の文書番号 docsdata_indexを作成
    docsdata_index = []
    for d in range(D):
        d_index = list(np.repeat(d, N[d]))
        for d_i in d_index:
            docsdata_index.append(d_i)
    # 混合ユニグラムモデル
    # トピック数  
    K = 3
    print("K=",K)
    # ハイパーパラメータ
    alpha = np.ones(K)*0.8
    beta = np.ones(V)*0.8
    # モデルの定義
    with pm.Model() as mixture_unigram_model:
        # トピックの分布
        theta = pm.Dirichlet("theta", a=alpha)
        # 単語の分布（K個トピックを全部）
        phi = pm.Dirichlet("phi", a=beta, shape=(K, V))
        # 文書のトピック
        z = pm.Categorical("z", p=theta, shape=D)
        # 文書内の単語
        w = pm.Categorical("w", p=phi[z][docsdata_index], observed=docsdata)
    # モデルの利用
    with mixture_unigram_model:
        idata = pm.sample(draws=5000, tune=1000, chains=4, progressbar=True)
        summary_idata = az.summary(idata, var_names=["theta", "phi"], kind="stats")
        print("サマリー：")
        print(summary_idata)
    # idataを保存
    with open("tpm-ch6ex2new-idata.pickle", "wb") as pklout:
        pickle.dump(idata, pklout, )
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")