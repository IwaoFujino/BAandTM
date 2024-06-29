# 20newsgroupsの単語集合にトピックモデル(LDA法)を適用、結果をpickleファイルに保存
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import datetime

# メイン関数
def main():
    # 20newsgroupsデータセットの単語集合を読み込む
    print("20newsgroupsデータセットの単語集合を読み込む...")
    with open("20newsgroups-docsdata.pickle", "rb") as f:
        docsdata = pickle.load(f)
    # 文書内では、単語別、出現回数をまとめる
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    docscount = count_vectorizer.fit_transform(docsdata)
    print("配列docscountのサイズ=", docscount.shape)
    vocabulary = count_vectorizer.get_feature_names_out()
    print("単語の総数 =", len(vocabulary))
    # トピックの抽出結果を保存する
    vocabfile = "20newsgroups-vocabulary.pickle"
    with open(vocabfile, "wb") as f1:
        pickle.dump(vocabulary, f1)
    print("単語帳を"+vocabfile+"に保存しました。")
    # トピックモデルを適用 
    nn_topics = 20
    print("トピックモデルを単語集合に適用する ...")
    lda = LatentDirichletAllocation(n_components=nn_topics)
    lda.fit(docscount)
    print("トピック単語行列を作成する ...")
    topics = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    print("トピック単語行列のサイズ =", topics.shape)
    # トピックの抽出結果を保存する
    topicfile="20newsgroups-topics.pickle"
    with open(topicfile, "wb") as f2:
        pickle.dump(topics, f2)
    print("トピック単語行列を"+topicfile+"に保存しました。")
    return

# ここから実行
if __name__ == "__main__":
	start_time = datetime.datetime.now()
	main()
	end_time = datetime.datetime.now()
	elapsed_time=end_time-start_time
	print("実行時間：", elapsed_time)