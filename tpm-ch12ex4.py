# 例題12.3で作ったデータセットにトピックモデル（LDA）を適用する。
# その結果を保存する
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import datetime
import pickle

# パラメータの設定
num_topics = 20

# メイン関数
def main():
    nn = 16
    print("nn=", nn)
    codebooklen = nn*nn*nn*nn
    print("コードブックの長さ =", codebooklen )
    print( "トピックの数=", num_topics )
    # データを読み込む
    print( "コードの文書集合を読み込む ..." )
    docsfile = "./aisdata/docsdata"+str(codebooklen)+".pickle"
    with open(docsfile, mode="rb") as fin:
            documents = pickle.load(fin)
    # トピックモデルを適用する
    print( "トピックモデル ..." )
  # 単語辞書、コーパスを作成、保存
    dictionary = Dictionary(documents)
    dictfile = "./aistopics/AIS-documents-dictionary.pickle"
    with open(dictfile, "wb") as f:
        pickle.dump(dictionary, f)
    print("単語辞書を"+dictfile+"に保存しました。")
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    corpfile = "./aistopics/AIS-documents-corpus.pickle"
    with open(corpfile, "wb") as f:
        pickle.dump(corpus, f)
    print("コーパスを"+corpfile+"に保存しました。")
    # トピックモデル(LDA法)
    print("トピックモデルを適用 ...")
    temp = dictionary[0]  # This is only to "load" the dictionary.
    model = LdaModel(
        corpus=corpus,
        id2word = dictionary,
        num_topics=num_topics,
        alpha="auto",
        eta="auto",
        iterations=400,
        passes=10,
        eval_every=1,
        random_state=0 # 再現性のため
    )
    # 評価：パープレシティ
    print("評価：トピック数", num_topics, "パープレシティ:", -model.log_perplexity(corpus)) 
    # モデルを保存
    modelfile = "./aistopics/topics"+str(num_topics)+".model"
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
