# 一様分布のヒストグラム
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

# 一様分布で乱数を生成
# 最小値=loc，最大値=loc+scale, 個数（サンプリング回数)=size
#rvs = stats.uniform.rvs(loc=10, scale=50, size=10000)
dist = stats.uniform(loc=10, scale=50)
rvs = dist.rvs(size=10000) 
# ヒストグラム
plt.hist(rvs, bins=10, alpha=0.3, ec='blue')
plt.xlabel("階級", fontsize=13)
plt.ylabel("出現回数", fontsize=13)
figfile="tpm-ch1ex1-fig1.png"
plt.savefig(figfile)
print("グラフを"+figfile+"に保存しました。")