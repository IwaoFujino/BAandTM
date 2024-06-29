# Twitterデータから著者トピックモデルのためのデータセットを作成する
import MeCab
import datetime
import pickle

# 形態素解析器
tagger = MeCab.Tagger()

# メイン関数
def main():
    tweetfile =  "./twittertext/twitter-authortext-20200313.txt"
    f = open(tweetfile, "r", encoding="utf8")
    docsauthor = []
    docsdata = []
    tweetcnt = 0
    minwords = 10
    for line in f:
        #print(line)
        try:
            linedic = eval(line)
        except (NameError, ValueError, SyntaxError):
            pass
        author = linedic["author"]
        tweettext = linedic["text"]
        docwords = []
        node = tagger.parseToNode(tweettext)
        while node:
            # 形態素属性を分割してリストに入れる
            node_features = node.feature.split(",")
            if node_features[0] == "名詞" and (node_features[1] == "一般" or node_features[1] == "固有名詞"):
                if node_features[6] != "*": # 見出し形のないものを除外
                    docwords.append(node_features[6]) # 見出し形を追加する
            node = node.next
        if len(docwords) >= minwords: # 単語数未満の文書を除外
            docsauthor.append(author)
            docsdata.append(docwords)
        tweetcnt += 1
    print("処理したツイートの数 =", tweetcnt)
    print("データセットに保存したツイートの数 =", len(docsdata))
    # 処理の結果を保存する
    authorfile = "twitter-docsauthor.pickle"
    with open(authorfile, "wb") as f:
        pickle.dump(docsauthor, f)
    print("docsauthorを"+authorfile+"に保存しました。")
    datafile = "twitter-docsdata.pickle"
    with open(datafile, "wb") as f:
        pickle.dump(docsdata, f)
    print("単語集合を"+datafile+"に保存しました。")
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")