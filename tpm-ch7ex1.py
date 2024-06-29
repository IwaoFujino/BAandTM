# トピックモデルで文書集合を生成
import numpy as np
import pymc as pm
import pickle

# 単語帳
vocabulary = ["赤", "緑", "青", "黄", "プードル", "マルチーズ", "ブルドッグ", "チワワ", "ばら", "たんぽぽ", "すみれ", "あじさい"]
print("単語帳：")
print(vocabulary)
# 単語帳をファイルに保存
vocabfile = "tpm-ch7ex1-vocabulary.pickle"
with open(vocabfile, "wb") as pklout:
    pickle.dump(vocabulary, pklout)
print("単語帳を"+vocabfile+"に保存しました。")
# 文書数
nn = 20
print("文書数：")
print(nn)
# 文書内の単語数
docmm = np.random.randint(20, 30, size=nn)
print("各文書内の単語数：")
print(docmm)
print("総単語数:", np.sum(docmm))
# 文書ごとのトピック分布を設定
theta_docs = [[0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.0, 0.9, 0.1],
            [0.9, 0.1, 0.0],
            [0.1, 0.6, 0.3],
            [0.1, 0.1, 0.8],
            [0.1, 0.6, 0.3],
            [0.9, 0.0, 0.1],
            [1.0, 0.0, 0.0],
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
            [0.7, 0.3, 0.0],
            [0.9, 0.1, 0.0],
            [0.1, 0.0, 0.9],
            [0.1, 0.6, 0.3],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.8, 0.1]]
# 色、犬と花の三つのトピックの単語分布を設定
phi_color = [0.2, 0.3, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
phi_dog = [0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0]
phi_flower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1]
print("各トピックの単語分布：")
print("phi_color=", phi_color)
print("phi_dog=", phi_dog)
print("phi_flower=",phi_flower)
# トピックのカテゴリ分布
distributions = [pm.Categorical.dist(p=phi_color),
                 pm.Categorical.dist(p=phi_dog),
                 pm.Categorical.dist(p=phi_flower)]
# 各文書の単語を生成            
documents = []
for d in range(nn):
    words = []
    for n in range(docmm[d]):
        zdn = pm.draw(pm.Categorical.dist(p=theta_docs[d]))
        wdn = pm.draw(distributions[zdn])
        words.append(vocabulary[wdn])
    doc = " ".join(words) # 2重引用符の間には半角の空白
    documents.append(doc)
print("作成した文書集合：")
for docno, doc in enumerate(documents):
    print("docno =", docno)
    print(doc)
docsfile = "tpm-ch7ex1-documents.pickle"
with open(docsfile, "wb") as fout:
    pickle.dump(documents, fout)
print("文書を"+docsfile+"に保存しました。")