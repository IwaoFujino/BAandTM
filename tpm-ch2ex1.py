# Pandas: 2重リストからデータフレームを作成
import pandas as pd

seiseki = [[89, 98, 81],
        [37, 67, 68],
        [78, 91, 92],
        [91, 90, 91],
        [67, 78, 89]]
seito = ["山田", "佐藤", "鈴木", "山下", "田中"]
kyouka = ["国語", "算数", "理科"]
# データフレームの作成、ラベルなし
seisekidf1 = pd.DataFrame(seiseki)
print("全教科の成績（成績のみ）：")
print(seisekidf1)
print("個別要素（3行、1列)=", seisekidf1.iloc[3,1])
# データフレームの作成、ラベルあり
seisekidf2 = pd.DataFrame(seiseki, columns=kyouka, index=seito)
print("全教科の成績（列ラベルと行ラベルを追加済み）：")
print(seisekidf2)
print("個別要素（山下、算数)=", seisekidf2.loc["山下", "算数"])
print("個別要素（3行、1列)=", seisekidf2.iloc[3, 1])
