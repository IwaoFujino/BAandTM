# ディリクレ分布
import numpy as np
from scipy import stats

# パラメータ
beta = [1, 2, 2, 1]
step = 0.25
# 確率変数のデータ
xvec = []
for i in np.arange(0, 1+step, step):
    # print("i=", i)
    for j in np.arange(0, 1+step, step):
        for k in np.arange(0, 1+step, step):
            for l in np.arange(0, 1+step, step):
                if i+j+k+l==1:
                    xvec.append([i, j, k, l])
# 確率分布密度関数
rv_pdf = []
for x in xvec:
    pdf = stats.dirichlet.pdf(x, beta)
    rv_pdf.append(pdf)
# 確率変数と確率分布密度関数を表示
print("---------------------")
for x, p in zip(xvec, rv_pdf):
    print("x=", x, "\tP=", f'{p:6.3f}')
print("---------------------")
print("データ数=",len(xvec))
print("実行が完了しました。")