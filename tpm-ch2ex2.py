# Pandas: ディクショナリからデータフレームを作成
import pandas as pd

seiseki = {"名前":["山田", "佐藤", "鈴木", "山下", "田中"],
            "国語": [89, 37, 78, 91, 67],
            "算数": [98, 67, 91, 90, 78],
            "理科": [81, 68, 92, 91, 89]}
seisekidf = pd.DataFrame(seiseki)
print("全教科の成績：")
print(seisekidf)
kokugodf = pd.DataFrame(seiseki, columns=["名前", "国語"])
print("国語の成績：")
print(kokugodf)
sansudf = pd.DataFrame(seiseki, columns=["名前", "算数"])
print("算数の成績：")
print(sansudf)
rikadf = pd.DataFrame(seiseki, columns=["名前", "理科"])
print("理科の成績：")
print(rikadf)
