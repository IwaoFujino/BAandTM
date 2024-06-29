# コサイン類似度
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 文書別の単語リスト
doc0 = ["野球", "大谷", "大谷", "優勝"] 
doc1 = ["野球", "野球", "野球", "試合", "試合","大谷"] 
doc2 = ["試合", "優勝", "優勝"] 
doc3 = ["試合", "試合", "試合", "大谷", "大谷", "優勝"] 
doc4 = ["試合", "試合", "大谷", "大谷", "大谷"] 
doc0str = " ".join(doc0)
doc1str = " ".join(doc1)
doc2str = " ".join(doc2)
doc3str = " ".join(doc4)
doc4str = " ".join(doc3)
docswords = [doc0str, doc1str, doc2str, doc3str, doc4str]
# TFIDF vectorizerを作成
vectorizer = TfidfVectorizer(norm=None, use_idf=True, smooth_idf=False)
vectors = vectorizer.fit_transform(docswords)
#単語リストを作成
wordlist = vectorizer.get_feature_names_out()
# TF-IDF文書単語行列
nn, mm = vectors.shape
# 文書間のコサイン類似度
print("文書間のコサイン類似度：")
result = cosine_similarity(vectors)
print(end="\t")
for n in range(nn):
    print("  doc=%d " % n, end=" ")
print()
for m in range(nn):
    print("doc=%d " % m, end="\t")
    for n in range(nn):
        print("%8.4f" % result[n, m], end="")
    print()
