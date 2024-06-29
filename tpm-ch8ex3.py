# 英語文書の形態素解析：名詞を抽出、見出し語に変換
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
with open("alice.txt","r", encoding="utf-8") as f:
    text = f.read()
print("文書:")
print(text)
words = nltk.word_tokenize(text)
print("単語:")
print(words)
wordtags = nltk.pos_tag(words)
docdata = []
for word, tag in wordtags:
    print(word, ">>", tag)
    if tag=='NN'or tag=='NNP' or tag=='NNS'or tag=='NNPS':
        docdata.append(lemmatizer.lemmatize(word, 'n'))
print("名詞リスト:")
print(docdata)