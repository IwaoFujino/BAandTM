# 画像の読み込みと画像の表示
import matplotlib.pyplot as plt

image = plt.imread("image_0001.jpg")
print("画像のサイズ", image.shape)
print("画像の行列のデータ")
print("R層：")
print(image[:, :, 0])
print("G層：")
print(image[:, :, 1])
print("B層：")
print(image[:, :, 2])
im = plt.imshow(image)
plt.show()