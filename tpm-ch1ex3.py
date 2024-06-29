# ベータ分布
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

# パラメータ
alpha = 0.5
beta = 0.5
# 確率密度関数
x = np.linspace(0, 1, 101)
rv_pdf = stats.expon.pdf(x, alpha, beta)
# 可視化
plt.plot(x, rv_pdf, '-', ms=8)
plt.grid()
plt.xlabel("確率変数x", fontsize=13)
plt.ylabel("確率密度関数", fontsize=13)
figfile="tpm-ch1ex3-fig1.png"
plt.savefig(figfile)
print("グラフを"+figfile+"に保存しました。")