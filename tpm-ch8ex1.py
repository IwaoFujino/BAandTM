# 20ニュースグループからデータとラベルを取得して表示する
from sklearn.datasets import fetch_20newsgroups

print("20ニュースグループデータセットを読み込む ...")
dataset = fetch_20newsgroups(subset="all", random_state=0, remove=("headers", "footers", "quotes"),)
print("データセットの属性名 =", dir(dataset))
for n in range(3):
    print("文書番号 =", n, "------------")
    print("カテゴリ =", dataset.target[n])
    print("テキスト =", dataset.data[n])
print("データセット内の文書の総数 =", len(dataset.data))
print("カテゴリの名称のリスト =", dataset.target_names)
print("データセット内のカテゴリの総数 =", len(dataset.target_names))
