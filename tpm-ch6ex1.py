# 混合ユニグラムモデル（三つのカテゴリ分布から合成）で文書集合を生成
import numpy as np
import pymc as pm
import pickle

# 単語帳
vocabulary = ["赤", "緑", "青", "黄", "プードル", "マルチーズ", "ブルドッグ", "チワワ", "ばら", "たんぽぽ", "すみれ", "あじさい"]
print("単語帳：")
print(vocabulary)
# 単語帳をファイルに保存
vocabfile = "tpm-ch6ex1-vocabulary.pickle"
with open(vocabfile, "wb") as pklout:
    pickle.dump(vocabulary, pklout)
print("単語帳を"+vocabfile+"に保存しました。")
# 文書数
nn = 50
print("文書数：")
print(nn)
# 文書のトピック分布を設定
theta_topic = [0.1, 0.3, 0.6]
distribution_topic = pm.Categorical.dist(p=theta_topic)
doc_topic = pm.draw(distribution_topic, draws=nn)
print("文書のトピック：")
print(doc_topic)
# 文書内の単語数
docmm = np.random.randint(20, 100, size=nn)
print("文書内の単語数：")
print(docmm)
print("総単語数:", np.sum(docmm))
# 色、犬と花の三つのトピックの単語分布を設定
phi_color = [0.2, 0.3, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
phi_dog = [0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0]
phi_flower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1]
print("単語の生成確率：")
print("phi_color =", phi_color)
print("phi_dog =", phi_dog)
print("phi_flower =",phi_flower)
# カテゴリ分布を使って文書を生成
distribution_color = pm.Categorical.dist(p=phi_color)
distribution_dog = pm.Categorical.dist(p=phi_dog)
distribution_flower = pm.Categorical.dist(p=phi_flower)
documents = []
for topic, mm in zip(doc_topic, docmm):
    if topic == 0:
        wordnos = pm.draw(distribution_color, draws=mm)
    if topic == 1:
        wordnos = pm.draw(distribution_dog, draws=mm)
    if topic == 2:
        wordnos = pm.draw(distribution_flower, draws=mm)
    words = []
    for wordno in wordnos:
        words.append(vocabulary[wordno])
    doc = " ".join(words) # 2重引用符の間には半角の空白
    documents.append(doc)
print("作成した文書集合：")
for docno, doc in enumerate(documents):
    print("docno =", docno)
    print(doc)
# 作成した文書集合をファイルに保存
docsfile = "tpm-ch6ex1-documents.pickle"
with open(docsfile, "wb") as pklout:
    pickle.dump(documents, pklout)
print("データを"+docsfile+"に保存しました。")