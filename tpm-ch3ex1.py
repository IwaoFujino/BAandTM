# ベルヌーイ分布からのサンプリング
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import datetime

plt.style.use("arviz-docgrid")

# main()関数
def main():
    distribution = pm.Bernoulli.dist(p=0.3)
    rvs = pm.draw(distribution, draws=100)
    print("サンプリングの結果：")
    print(rvs)
    print("データサイズ =", rvs.shape)
    rvs_df = pd.DataFrame(rvs)
    print("データフレームの最初の５個データ：")
    print(rvs_df.head(5))
    # サンプルデータのグラフ
    rvs_df.plot()
    plt.xlabel("サンプル番号")
    plt.ylabel("サンプル値")
    figfile = "tpm-ch3ex1-fig1.png"
    plt.savefig(figfile)
    print("グラフを"+figfile+"に保存しました。")
    # ヒストグラム
    rvs_df.hist(bins=20)
    plt.xlabel("階級")
    plt.ylabel("出現回数")
    figfile = "tpm-ch3ex1-fig2.png"
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