# 英語文書の形態素解析：単語を分割、タグを表示
import nltk

with open("alice.txt","r", encoding="utf-8") as f:
    text = f.read()
print("文書:")
print(text)
words = nltk.word_tokenize(text)
print("単語:")
print(words)
wordtags = nltk.pos_tag(words)
print("単語と品詞:")
for word, tag in (wordtags):
    print("単語=", word, "\t 品詞=", tag)