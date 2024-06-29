# 例題7.1で作ったデータセットにトピックモデル（LDA）を適用する。
# その結果を保存する
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pickle

# データセットを読み込む
print("データセットを読み込む...")
with open("tpm-ch7ex1x1-documents.pickle", "rb") as f:
    documents = pickle.load(f)
docsdata=[]
for doc in documents:
    docdata=doc.split()
    print(docdata)
    docsdata.append(docdata)
# 辞書、コーパスを作成
dictionary = Dictionary(docsdata)
print("単語帳：")
print(dictionary.token2id)
corpus = [dictionary.doc2bow(docdata) for docdata in docsdata]
print("コーパス：")
print(corpus)
# トピックモデル(LDA法)
print("トピックモデルを適用 ...")
temp = dictionary[0]
model = LdaModel(
    corpus=corpus,
    num_topics=3,
    alpha="auto",
    eta="auto",
    iterations=400,
    passes=10,
    eval_every=1,
    random_state=0 # 再現性のため
)
# トピック単語行列を表示
print("トピック単語行列を読み込む...")
topics = model.get_topics()
# 各トピックのトップワードを表示する
num_words = 5
for topicno, topic in enumerate(topics):
    print("トピック番号 =", topicno)
    idx = np.argsort(-topic)[:num_words]
    for id in idx:
        print(dictionary[id], "\t", np.round(topic[id], 6))
# 評価：パープレシティ
print("評価：パープレシティ: ", np.exp(-model.log_perplexity(corpus)))
