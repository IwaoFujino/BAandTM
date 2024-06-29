# ガンマ分布により世帯所得を作成、各種統計量とヒストグラム
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# パラメータ設定
b = 0.255
a = 5.643*b
# 厚生労働省のデータに基づき、ガンマ分布により世帯所得の標本を生成
rvs = stats.gamma.rvs(a, scale = 1.0/b, random_state=0, size=51914000)
shotoku = rvs/10.0 # 所得の単位を千万円に直す
#リストからシリーズを作成。
shotokusr = pd.Series(shotoku)
print("最初の10個データ:")
print(shotokusr.head(10))
print("最後の10個データ:")
print(shotokusr.tail(10))
# 平均値、標準偏差、中央値、下位10%分割点、上位10%分割点
print("平均値 =",f'{shotokusr.mean():.4f}千万円')
print("中央値 =",f'{shotokusr.median():.4f}千万円')
print("標準偏差 =", f'{shotokusr.std():.4f}千万円')
print("下位10パーセント分割点 =", f'{shotokusr.quantile(0.10):.4f}千万円')
print("上位10パーセント分割点 =", f'{shotokusr.quantile(0.90):.4f}千万円')
# ヒストグラム
plt.figure(figsize=(10, 7))
plt.hist(shotokusr, bins=100, range=(0, 5), alpha=0.6, ec="blue")
plt.xlabel("階級（千万円）", fontsize=16)
plt.ylabel("世帯数", fontsize=16)
plt.ticklabel_format(style="plain",axis="y")
plt.grid()
plt.savefig("tpm-ch2ex5-fig1.png")