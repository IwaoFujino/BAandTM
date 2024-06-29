# ベルヌーイ分布のパラメータの推定＋サマリーと事後分布グラフ
import pymc as pm
from scipy import stats
from matplotlib import pyplot as plt
import arviz as az
import datetime

plt.style.use("arviz-docgrid")

# メイン関数
def main():
    # 観測データを作成
    obstheta = 0.7
    obsmm = 1000
    obsdata = stats.bernoulli.rvs(p=obstheta, size=obsmm)
    print("観測データ：")
    print(obsdata)
    # ベイズ推定モデル
    with pm.Model() as model:
        # 事前分布：0 〜 1 の一様分布とする
        theta = pm.Uniform("theta", lower=0, upper=1)
        # 推定モデルはベルヌーイ分布とする
        distribution = pm.Bernoulli("distribution", p=theta, observed=obsdata)
    # 推定を実行し、事後分布からサンプルを得る
    with model:
        idata = pm.sample(10000)
        print(az.summary(idata))
        az.plot_trace(idata, compact=False)
        figfile = "tpm-ch3ex3-fig1.png"
        plt.savefig(figfile)
        print("グラフを"+figfile+"に保存しました。")
        az.plot_posterior(idata, hdi_prob=0.95)
        figfile = "tpm-ch3ex3-fig2.png"
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