# MeCab:固有名詞、名詞と動詞を抽出
import MeCab

tagger = MeCab.Tagger()
text = """日本国民は、正当に選挙された国会における代表者を通じて行動し、
われらとわれらの子孫のために、諸国民との協和による成果と、
わが国全土にわたって自由のもたらす恵沢を確保し、
政府の行為によって再び戦争の惨禍が起ることのないようにすることを決意し、
ここに主権が国民に存することを宣言し、この憲法を確定する。"""
print("日本語の文：")
print(text)
words=[]
node = tagger.parseToNode(text)
while node:
	# 形態素属性を分割して、名詞と動詞をリストに入れる
    node_features=node.feature.split(",")
    #print(node.surface, ">>>", node_features)
    if node_features[0]=="名詞" and (node_features[1] in ["一般", "固有名詞", "サ変接続", "形容動詞語幹"]):
        words.append(node_features[6])
    if node_features[0]=="動詞" and node_features[1]=="自立" and node_features[6]!="する":
        if node_features[6]!="する":
            words.append(node_features[6])
    node = node.next
print('名詞と動詞のリスト：')
print(words)
