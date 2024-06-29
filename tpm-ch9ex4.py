# トピック別の上位単語、文書別のトピックの割合を表示する
from gensim import  models
import pickle

# 保存済みのデータを読み込む
print("辞書とコーパスをロード...")
maxdocs = 100000
dictfile = "wiki"+str(maxdocs)+"-dictionary.pickle"
with open(dictfile, "rb") as f:
    dictionary = pickle.load(f)
corpfile = "wiki"+str(maxdocs)+"-corpus.pickle"
with open(corpfile, "rb") as f:
    corpus = pickle.load(f)
# 訓練済みのモデルをロードする
print("モデルをロード...")
num_topics = 20
modelfile = "wiki"+str(maxdocs)+"-lda"+str(num_topics)+".model"
model = models.ldamodel.LdaModel.load(modelfile)
# 各トピックの上位単語を取得する
print("トピック別の上位単語：")
for topicid in range(num_topics):
    topwords = model.get_topic_terms(topicid, topn=10)
    print("トピック番号 =", topicid, "----------")
    for wordid, prob in topwords:
        print(dictionary[wordid], "\t", prob.round(8))
print("文書別のトピックの割合：")
document_topics = model.get_document_topics(corpus)
for docid, doctopics in enumerate(document_topics[0:10]):
    print("文書番号 =", docid, "--------")
    topics=dict(doctopics)
    print("トピック   割  合", )
    for tpid, prop in sorted(topics.items(), key=lambda x : x[1], reverse=True):
        print(tpid, "\t", prop.round(8))
        