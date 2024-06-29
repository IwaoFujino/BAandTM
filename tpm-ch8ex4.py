# 20 news groupsから名詞を抽出して、文書ごとに文字列にまとめてリストに保存する。
# さらに、そのリストをpickleファイルに書き出す。
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
import pickle
import datetime

# 見出し語器
lemmatizer = WordNetLemmatizer()

#メイン関数
def main():
    # データセットを読み込む
    print("20ニュースグループデータセットを読み込む ...")
    dataset = fetch_20newsgroups(subset="all", random_state=0, remove=("headers", "footers", "quotes"))
    print("データセット内の文書の総数=", len(dataset.data))
    # 文書ごとに名詞を抽出する
    print("文書ごとに名詞を抽出する ...")
    docsdata = []
    for docno, text in enumerate(dataset.data):
        words = []
        wordtags = nltk.pos_tag(nltk.word_tokenize(text))
        for word, tag in wordtags:
            if tag=='NN'or tag=='NNP' or tag=='NNS'or tag=='NNPS':
                lemma = lemmatizer.lemmatize(word)
                if(len(lemma)>=3): # 単語は３文字以上
                    words.append(lemma)
        if (len(words)>=5): # 文書は５単語以上
            docdata = " ".join(words)
            docsdata.append(docdata)
    # 名詞の抽出結果を表示する
    print("抽出した名詞を表示する ...")
    for docno, docdata in enumerate(docsdata):
        print("文書番号 =", docno, "-----------------------")
        print("名詞リスト =", docdata)
    # 名詞の抽出結果を保存する
    docsfile = "20newsgroups-docsdata.pickle"
    with open(docsfile, "wb") as f:
        pickle.dump(docsdata, f)
    print("単語集合を"+docsfile+"に保存しました。")

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")