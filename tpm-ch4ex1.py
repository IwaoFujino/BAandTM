# MeCabによる単語の抽出と属性情報の分離
import MeCab

tagger = MeCab.Tagger()
text = "彼は３浪までして念願の東京大学法学部に合格できました。"
# 解析結果をオブジェクト変数nodeに入れる
node = tagger.parseToNode(text) 
while node:
    # node.surface 形態素の表記
    # node.feature 形態素の品詞、読みなどの属性情報
    print("単語の表記：", node.surface)
    print("単語の属性（文字列）：", node.feature)
    node_features = node.feature.split(",")
    print("単語の属性（リスト）：", node_features)
    node = node.next