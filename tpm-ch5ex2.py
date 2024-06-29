# 生成済み文書集合のロードと単語の出現頻度グラフ
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import pickle

# 文書集合のデータをロード
with open("tpm-ch5ex1-documents.pickle", "rb") as pklin:
    documents = pickle.load(pklin)
# 全部の文書の単語をリストに
allwords = " ".join(documents) # 2重引用符の間には半角の空白
allwords = allwords.split()
print(allwords)
wordsdf = pd.DataFrame(allwords)
wordsfrequency = wordsdf.value_counts(normalize=True, sort=True)
print("単語の出現頻度（降順）：")
print(wordsfrequency)
wordsfrequency.plot.barh(grid=True)
figfile = "tpm-ch5ex2-fig1.png"
plt.savefig(figfile)
print("グラフを"+figfile+"に保存しました。")