# 画像の読み込みと画像の表示
import matplotlib.pyplot as plt

image = plt.imread("image_0001.jpg")
print("元の画像のサイズ", image.shape)
print("画像のトリミング")
image = image[0:100, 0:150, :]
print("トリミング後の画像のサイズ", image.shape)
plt.imshow(image)
plt.savefig("trimmed-image.jpg")