# 画像から作成した文書集合にトピックモデルを適用する
# その結果を保存する
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pickle
import datetime

# パラメータ設定
num_topics = 20

#メイン関数
def main():
    # データセットを読み込む
    print("データセットを読み込む...")
    docsfile = "image-BovW-documents.pickle"
    with open(docsfile, "rb") as f:
        documents = pickle.load(f)
    # 単語辞書、コーパスを作成、保存
    dictionary = Dictionary(documents)
    dictfile = "image-BoVW-dictionary.pickle"
    with open(dictfile, "wb") as f:
        pickle.dump(dictionary, f)
    print("単語辞書を"+dictfile+"に保存しました。")
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    corpfile = "image-BoVW-corpus.pickle"
    with open(corpfile, "wb") as f:
        pickle.dump(corpus, f)
    print("コーパスを"+corpfile+"に保存しました。")
    # トピックモデル(LDA法)
    print("トピックモデルを適用 ...")
    temp = dictionary[0]
    model = LdaModel(
        corpus=corpus,
        num_topics=num_topics,
        alpha="auto",
        eta="auto",
        iterations=400,
        passes=10,
        eval_every=None,
        random_state=0 # 再現性のため
    )
    # 評価：パープレシティ
    print("評価：パープレシティ：", -model.log_perplexity(corpus))
    # モデルを保存
    modelfile = "image-BoVW-lda"+str(num_topics)+".model"
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