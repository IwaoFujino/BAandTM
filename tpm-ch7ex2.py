# トピックモデル
# ディリクレ分布を用いて、カテゴリ分布のパラメータを推定
# 推定結果をファイルに保存する
import numpy as np
import pickle
import pymc as pm
import arviz as az
import datetime

# メイン関数
def main():
    # 文書集合を読み込む
    with open("tpm-ch7ex1-documents.pickle", "rb") as pklin:
        documents = pickle.load(pklin)
    #print("文書集合の中身：")
    #print(documents)
    #単語リストを作成
    with open("tpm-ch7ex1-vocabulary.pickle", "rb") as pklin:
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
        docdata = []
        for word in words:
            wordid = vocabulary.index(word)
            docdata.append(wordid)
        docsdata.append(docdata)
    # 文書データを表示
    print("文書データ：")
    print(docsdata)
    print("文書数D =",D)
    print("各文書の単語数N =",N)
    print("単語帳の単語数V =",V)
    # トピックモデル
    # トピック数   
    K = 3
    print("トピック数K =", K)
    # ハイパーパラメータ
    alpha = np.ones([D, K])*0.8 
    beta = np.ones([K, V])*0.8 
    # モデルの定義
    with pm.Model() as topic_model:
        # トピックの分布
        theta = pm.Dirichlet('theta', a=alpha, shape=(D, K))
        # 単語の分布（K個トピックを全部）
        phi = pm.Dirichlet('phi', a=beta, shape=(K, V))
        # 単語のトピック
        z = []
        for d in range(D):
            z1 = pm.Categorical("z_{}".format(d), p = theta[d], shape=int(N[d]))
            z.append(z1)
        print("zのサイズ =", len(z))
        # 文書内の単語
        w = []
        for d in range(D):
            w1 = pm.Categorical('w_{}'.format(d), p=phi[z[d]], observed = docsdata[d], shape=int(N[d])) 
            w.append(w1)
        print("wのサイズ =", len(w))
    # モデルの利用
    with topic_model:
        idata = pm.sample(draws=5000, tune=1000, chains=4, progressbar=True)
        summary_theta = az.summary(idata, var_names=["theta"], kind="stats")
        print("thetaのサマリー：")
        print(summary_theta)
        summary_phi = az.summary(idata, var_names=["phi"], kind="stats")
        print("phiのサマリー：")
        print(summary_phi)
    # idataを保存
    with open("tpm-ch7ex2-idata.pickle", "wb") as pklout:
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