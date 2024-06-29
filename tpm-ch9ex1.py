# wikipediaのjsonデータファイルから記事タイトルと記事内の単語を取得、保存する
import datetime
import MeCab
from gensim import utils
import json
import pickle

# 形態素解析器
tagger = MeCab.Tagger()

# wiki記事のjsonファイルから記事タイトルと記事内の単語を取得
def getwikidata(jsonfile, maxdocs, minwords):
    # wiki記事から名詞を抽出
    titles = []
    documents = []
    doccnt=0
    print("wikiの記事を一つずつ処理する ...")
    with utils.open(jsonfile, "rb") as f:
        for line in f:
            article = json.loads(line)
            if doccnt % 1000 == 0:
                print("処理中... 文書番号 =", doccnt, "タイトル =", article["title"])
            linetext=[]
            for section_title, section_text in zip(article["section_titles"], article["section_texts"]):
                linetext.append(section_title)
                linetext.append(section_text)
            doctext =" ".join(linetext)
            docwords=[]
            node = tagger.parseToNode(doctext)
            while node:
                # 形態素属性を分割してリストに入れる
                node_features=node.feature.split(",")
                if node_features[0]=="名詞" and (node_features[1]=="一般" or node_features[1]=="固有名詞"):
                    if node_features[6] != "*": # 原形のないものを除外
                        docwords.append(node_features[6]) # 原形を追加する
                node = node.next
            if len(docwords) >= minwords: # 単語数未満の文書を除外
                titles.append(article["title"])
                documents.append(docwords)
                doccnt += 1
            if doccnt == maxdocs:
                break
    # タイトルと単語数を表示。文書数を表示
    print("処理の結果...")
    wordcnt=0
    for title, docwords in zip(titles, documents):
        print("タイトル =", title, "単語数 =", len(docwords))
        wordcnt += len(docwords)
    print("文書の総数 =", len(titles))
    print("単語の総数 =", wordcnt)
    return titles, documents

# メイン関数
def main():
    jsonfile = "jawiki-latest.json.gz"
    maxdocs = 100000 # 取得する記事の最大数。10万記事で7分ぐらいかかる。
    print("取得記事数", maxdocs)
    minwords = 200 # １記事の最小単語数
    titles, documents = getwikidata(jsonfile, maxdocs, minwords)
    # 処理の結果を保存する
    titlefile="wiki"+str(maxdocs)+"-titles.pickle"
    with open(titlefile, "wb") as f:
        pickle.dump(titles, f)
    print("タイトルを"+titlefile+"に保存しました。")
    docfile="wiki"+str(maxdocs)+"-documents.pickle"
    with open(docfile, "wb") as f:
        pickle.dump(documents, f)
    print("単語集合を"+docfile+"に保存しました。")
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")
    