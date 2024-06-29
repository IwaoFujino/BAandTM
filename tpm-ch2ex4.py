# Pandas: 各種統計量の計算と棒グラフ
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# csvファイルからデータを読み込んで、データフレームを作成する。
seisekidf = pd.read_csv("./seiseki-sum.csv", index_col=0)
print("全教科の成績：")
print(seisekidf)
print("合計点の列を表示")
print(seisekidf["合計"])
print("合計点を表示(合計点の昇順)")
print(seisekidf.sort_values("合計"))
# 統計量の計算
print("平均値=", f'{seisekidf["合計"].mean():.2f}')
print("中央値=", f'{seisekidf["合計"].median():.2f}')
print("標準偏差=", f'{seisekidf["合計"].std():.2f}')
print("第1四分位数=", f'{seisekidf["合計"].quantile(0.25):.2f}')
print("第3四分位数=", f'{seisekidf["合計"].quantile(0.75):.2f}')
print("四分位数範囲=", f'{seisekidf["合計"].quantile(0.75)-seisekidf["合計"].quantile(0.25):.2f}')
# 棒グラフ
plt.figure()
plt.bar(seisekidf["合計"].index, seisekidf["合計"].values) # 縦棒の棒グラフ
plt.hlines(seisekidf["合計"].mean(), -1, 5, colors="red", linestyles="solid", label="平均値")
plt.hlines(seisekidf["合計"].quantile(0.25), -1, 5, colors="yellow", linestyles="dashed", label="第1四分位数")
plt.hlines(seisekidf["合計"].quantile(0.75), -1, 5, colors="yellow", linestyles="dashed", label="第3四分位数")
plt.xlabel("名前", fontsize=16)
plt.ylabel("合計点", fontsize=16)
plt.legend()
plt.grid()
plt.savefig("tpm-ch2ex4-fig1.png")
