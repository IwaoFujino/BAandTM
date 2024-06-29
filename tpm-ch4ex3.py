# 文書集合のTFIDF単語文書行列
from sklearn.feature_extraction.text import TfidfVectorizer

# 文書別の単語リスト
doc0 = ["野球", "大谷", "大谷", "優勝"] 
doc1 = ["野球", "野球", "野球", "試合", "試合","大谷"] 
doc2 = ["試合", "優勝", "優勝"] 
doc3 = ["試合", "試合", "試合", "大谷", "大谷", "優勝"] 
doc4 = ["試合", "試合", "大谷", "大谷", "大谷"] 
doc0str = " ".join(doc0) # 2重引用符の間には半角の空白
doc1str = " ".join(doc1) # 2重引用符の間には半角の空白
doc2str = " ".join(doc2) # 2重引用符の間には半角の空白
doc3str = " ".join(doc3) # 2重引用符の間には半角の空白
doc4str = " ".join(doc4) # 2重引用符の間には半角の空白
docswords = [doc0str, doc1str, doc2str, doc3str, doc4str]
print("文書別の単語：")
print(docswords)
# TFIDF vectorizerを作成
vectorizer = TfidfVectorizer(norm=None, use_idf=True, smooth_idf=False)
vectors = vectorizer.fit_transform(docswords)
# 単語リストを作成
wordlist = vectorizer.get_feature_names_out()
# ベクトルの転置
vectors = vectors.T
# TFIDF文書単語行列
mm, nn = vectors.shape
print("TF-IDFの行列のサイズ：", vectors.shape)
print("TF-IDFの行列：")
print("%s" % "単語", end="\t")
for n in range(nn):
    print("  doc=%d " % n, end=" ")
print()
for m in range(mm):
    print("%s" % wordlist[m], end="\t")
    for n in range(nn):
        print("%8.4f" % vectors[m, n], end=" ")
    print()
