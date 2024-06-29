# ベクトル量子化
import numpy as np
from matplotlib import pyplot as plt
import scipy.cluster
import japanize_matplotlib

# ベクトルデータ
data = [
[1, 2],
[3, 1],
[2, 3],
[3, 6],
[4, 6],
[7, 2],
[7, 4],
]
npdata = np.array(data, dtype=float)
num_codes = 3
# コードブックを作成
codebook, distortion = scipy.cluster.vq.kmeans(npdata, num_codes, iter=30, thresh=1e-06)
print("コードブック：")
print(codebook)
# 各データをセントロイドに分類
codes, dists = scipy.cluster.vq.vq(npdata, codebook)
print("データのコード：")
print("ベクトル       コード   セントロイド")
for data, code in zip(npdata, codes):
    print(data, "\t", code, "\t", codebook[code])
# コード 0 の描画
ldata = npdata[codes == 0]
plt.scatter(ldata[:, 0], ldata[:, 1], marker='s', color='green')
plt.scatter(codebook[0, 0], codebook[0,1], marker="o", color="green", label="コード0のセントロイド")
# コード 1 の描画
ldata = npdata[codes == 1]
plt.scatter(ldata[:, 0], ldata[:, 1], marker='x', color='red')
plt.scatter(codebook[1, 0], codebook[1, 1], marker="o", color="red", label="コード1のセントロイド")
# コード 2 の描画
ldata = npdata[codes == 2]
plt.scatter(ldata[:, 0], ldata[:, 1], marker='^', color='blue')
plt.scatter(codebook[2, 0], codebook[2, 1], marker="o", color="blue", label="コード2のセントロイド")
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.legend()
plt.grid()
plt.savefig("tpm-ch11ex3-fig1.png")