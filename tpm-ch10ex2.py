# 著者トピックモデル
# 推定結果をファイルに保存する
import numpy as np
import datetime
import pickle
import pymc as pm
import arviz as az

# メイン関数
def main():
    # 文書著者を読み込む
    S = 5
    print("著者数 =", S)
    authorfile = "tpm-ch10ex1-docsauthor.pickle"
    with open(authorfile, "rb") as pklin:
        authorsdata = pickle.load(pklin)
    # 文書集合を読み込む
    docsfile = "tpm-ch10ex1-documents.pickle"
    with open(docsfile, "rb") as pklin:
        documents = pickle.load(pklin)
    #単語リストを作成
    dictfile = "tpm-ch10ex1-vocabulary.pickle"
    with open(dictfile, "rb") as pklin:
        vocabulary = pickle.load(pklin)
    print("単語辞書：")
    print(vocabulary)
    V = len(vocabulary)
    #単語を単語番号に変換
    docsdata = []
    D = len(documents)
    N = []
    for doc in documents:
        words = doc.split()
        N.append(len(words))
        docdata=[]
        for word in words:
            wordid = vocabulary.index(word)
            docdata.append(wordid)
        docsdata.append(docdata)
    # 文書データを表示
    print("文書データ：")
    print(docsdata)
    print("文書数D =",D)
    print("各文書の単語数N =",N)
    print("単語辞書の単語数V =",V)
    # 著者トピックモデル
    # トピック数   
    K = 3
    print("トピック数K=", K)
    # ハイパーパラメータ
    alpha = np.ones([S, K])*0.8
    beta = np.ones([K, V])*0.8
    gamma = np.ones(S)/S
    # モデルの定義
    with pm.Model() as author_topic_model:
        # 著者のトピックの分布(S人の著者を全部)
        theta   = pm.Dirichlet('theta', a=alpha, shape=(S, K))
        # トピックの単語分布（K個のトピックを全部）
        phi = pm.Dirichlet('phi', a=beta, shape=(K, V))
        w=[]
        for d in range(D):
            # 著者を選択
            yd = pm.Categorical("y_{}".format(d), p = gamma, observed = authorsdata[d], shape=int(N[d]))
            # 単語のトピック
            zd = pm.Categorical("z_{}".format(d), p = theta[yd], shape=int(N[d]))
            # 文書内の単語
            wd = pm.Categorical('w_{}'.format(d), p=phi[zd], observed = docsdata[d], shape=int(N[d]))  
            w.append(wd)
        print("wのサイズ =", len(w))
    # モデルの利用
    with author_topic_model:
        idata = pm.sample(draws=5000, tune=1000, chains=4, progressbar=True)
        # idataを保存
        with open("tpm-ch10ex2-idata.pickle", "wb") as fout:
            pickle.dump(idata, fout, )
        phi_posterior = az.extract(idata, var_names=["phi"])
        kk, mm, nn = phi_posterior.shape
        phis = phi_posterior.values[:,:,nn-1]
        print("トピック別の単語と確率（確率の降順）：")
        for tpno, tpprob in enumerate(phis):
            print("トピック=", tpno)
            wordidx = np.argsort(-tpprob)
            for id in wordidx:
                id = int(id)
                print(vocabulary[id], "\t\t", tpprob[id].round(6))
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")