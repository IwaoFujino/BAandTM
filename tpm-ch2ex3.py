# Pandas: CSVファイルの入力と出力
import pandas as pd

# csvファイルからデータを読み込んで、データフレームを作成
seisekidf = pd.read_csv("./seiseki.csv", index_col=0)
print("全教科の成績：")
print(seisekidf)
print("インデックス（行名）を表示")
print(seisekidf.index)
print("項目（列名）を表示")
print(seisekidf.columns)
print("値を表示")
print(seisekidf.values)
print("行の合計を表示")
print(seisekidf.sum(axis=1))
print("全教科の成績（合計の列を追加）：")
seisekidf["合計"] = seisekidf.sum(axis=1)
print(seisekidf)
seisekidf.to_csv("./seiseki-sum.csv")