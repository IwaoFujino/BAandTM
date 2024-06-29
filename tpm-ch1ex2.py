# 二項分布の確率質量関数のグラフを作成
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

# 確率質量関数
phi = 0.3
nn = 10
x = np.arange(0, nn+1) 
rv_pmf = stats.binom.pmf(x, nn, phi)
# グラフ作成
plt.plot(x, rv_pmf, color='blue', marker='o', linestyle='', markersize=8)
plt.vlines(x, 0, rv_pmf, colors='b', linewidth=3, alpha=0.5)
plt.grid()
plt.xlim(-1, nn+1) # x軸目盛
plt.ylim(0, 1) # y軸目盛
plt.xlabel("確率変数x", fontsize=13)
plt.ylabel("確率質量関数", fontsize=13)
figfile="tpm-ch1ex2-fig1.png"
plt.savefig(figfile)
print("グラフを"+figfile+"に保存しました。")