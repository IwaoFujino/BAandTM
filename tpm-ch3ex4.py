# 正規分布のパラメータの推定＋サマリーと事後分布密度グラフ
import numpy as np
import pymc as pm
from scipy import stats
from matplotlib import pyplot as plt
import arviz as az
import datetime

plt.style.use('arviz-docgrid')

# メイン関数
def main():
    # 観測データを作成
    print("観測データを作成：")
    obsmu = 0.0
    obssigma = 1.0
    obsnn = 100
    print("平均値=", obsmu)
    print("標準偏差=", obssigma)
    print("サンプル数=", obsnn)
    print("---------------")
    obsdata = stats.norm.rvs(obsmu, obssigma, size=obsnn)
    # ベイズ推定モデル
    with pm.Model() as model:
        # mu, sigmaの事前分布は一様分布
        mu = pm.Uniform("mu", lower=-10, upper=10)
        sigma = pm.Uniform("sigma", lower=0, upper=10)
        distribution = pm.Normal("distribution", mu=mu, sigma=sigma, observed=np.array(obsdata)) 
    # 推定を実行し、事後分布からサンプルを得る
    with model:
        idata = pm.sample(10000)
        print(az.summary(idata))
        az.plot_trace(idata, compact=False)
        figfile = "tpm-ch3ex4-fig1.png"
        plt.savefig(figfile)
        print("グラフを"+figfile+"に保存しました。")
        az.plot_posterior(idata, hdi_prob=0.95)
        figfile = "tpm-ch3ex4-fig2.png"
        plt.savefig(figfile)
        print("グラフを"+figfile+"に保存しました。")
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")