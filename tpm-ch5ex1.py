# カテゴリ分布を用いて、文書（の単語）を生成する
# 文書集合（複数の文書）を生成する
# pickleファイルに保存する
import numpy as np
import pymc as pm
import pickle

# 単語帳(全部で10個)
vocabulary = ["赤", "白", "黄", "緑", "紫", "青", "黒", "オレンジ", "ピンク", "茶"]
vocabfile = "tpm-ch5ex1-vocabulary.pickle"
with open(vocabfile, "wb") as pklout:
    pickle.dump(vocabulary, pklout)
print("単語帳を"+vocabfile+"に保存しました。")
# 文書内の単語数のリスト
nn = 20
docmm = np.random.randint(20, 100, size=nn)
print("文書内の単語数：")
print(docmm)
theta = [0.03, 0.1, 0.3, 0.1, 0.05, 0.2, 0.03, 0.05, 0.04, 0.1]
print("単語の生成確率(降順）：")
thetaidx = np.argsort(-np.array(theta))
for id in thetaidx:
    print(vocabulary[id], theta[id])
# カテゴリ分布を使って単語を生成
distribution = pm.Categorical.dist(p=theta)
documents = []
for mm in docmm:
    wordnos = pm.draw(distribution, draws=mm)
    words = []
    for wordno in wordnos:
        words.append(vocabulary[wordno])
    document = " ".join(words) # 2重引用符の間には半角の空白
    #print(document)
    documents.append(document)
print("作成した文書：")
for docno, doc in enumerate(documents):
    print("docno=", docno)
    print(doc)
docsfile = "tpm-ch5ex1-documents.pickle"
with open(docsfile, "wb") as pklout:
    pickle.dump(documents, pklout)
print("データを"+docsfile+"に保存しました。")
