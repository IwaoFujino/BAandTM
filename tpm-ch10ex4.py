# Twitterデータセットに、著者トピックモデルを適用する。
# トピックをコヒーレンスの順に表示する
from gensim.models import AuthorTopicModel
from gensim.corpora import Dictionary
import pickle
import datetime

# メイン関数
def main():
    # トピック数
    num_topics = 20
    print("トピックの数 =", num_topics)
    # データセットを読み込む
    authorfile = "twitter-docsauthor.pickle"
    with open(authorfile, "rb") as fin:
        docsauthor = pickle.load(fin)
    print("著者の数 =", len(list(set(docsauthor))))
    #print(docsauthor)
    # 文書集合を読み込む
    docfile = "twitter-docsdata.pickle"
    with open(docfile, "rb") as fin:
        docsdata = pickle.load(fin)
    print("文書の数 =", len(docsdata))
    # 文書の著者を作成
    author2doc={}
    for docid, author in enumerate(docsauthor):
        if author in author2doc.keys():
            author2doc[author].append(docid)
        else:
            author2doc[author]=[docid]
    # 辞書、コーパスを作成
    print("単語辞書を作成 ...")
    dictionary = Dictionary(docsdata)
    dictfile = "twitter-dictionary.pickle"
    with open(dictfile, "wb") as f:
        pickle.dump(dictionary, f)
    print("単語辞書を"+dictfile+"に保存しました。")
    print("コーパスを作成 ...")
    corpus = [dictionary.doc2bow(docdata) for docdata in docsdata]
    corpfile = "twitter-corpus.pickle"
    with open(corpfile, "wb") as f:
        pickle.dump(corpus, f)
    print("コーパスを"+corpfile+"に保存しました。")
    #print(corpus)
    # 著者トピックモデルを適用する
    print("著者トピックモデルを適用 ...")
    model = AuthorTopicModel(
        corpus=corpus,
        author2doc=author2doc,
        id2word=dictionary,
        num_topics=num_topics,
        iterations=400,
        passes=10,
        eval_every=None,
        random_state=0 # 再現性のため
    )
    # 各トピックの上位単語を表示する
    toptopics = model.top_topics(corpus, topn=10)
    print("コヒーレンスの降順でソートしたトピック：")
    for topicno, topic in enumerate(toptopics):
        print("-----", topicno, "番のトピック -----")
        for tp in topic[0]:
            print(tp[1], tp[0].round(6))
        print("コヒーレンス =", topic[1].round(6))
    # モデルを保存
    modelfile = "./twitter-author-TPM"+str(num_topics)+".model"
    model.save(modelfile)
    print("モデルを" + modelfile + "に保存しました。")
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")