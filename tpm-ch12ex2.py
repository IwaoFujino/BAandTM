# 例題12.2: AISデータを読み込んで、航跡を地図に描く
import datetime
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
import japanize_matplotlib

# ブレスト湾外の海域
west = -6.00 # 経度（西（左））
east = -3.00 # 経度（東（右））
south = 46.75 # 緯度（南（下））　
north = 48.75 # 緯度（北（上））
lonscale = 0.50
latscale = 0.25

# データファイルを読み込む
def readsailaisdata():
    with open("./aisdata/sailsbysailid.pickle", mode="rb") as fin:
        sailsbysailid = pickle.load(fin)
    for sailid, saildata in sailsbysailid.items():
        print("航走ID =", sailid)
        print("航跡データ =", saildata)
    return sailsbysailid

# 航跡を描く
def drawtrajectory(sailid,sailaisdata):
    # データを用意する
    lons = sailaisdata[:, 0]
    lats = sailaisdata[:, 1]
    headings = sailaisdata[:, 2]
    speeds = sailaisdata[:, 3]
    r0 = 0.06
    sita0 = (90.0-headings)*np.pi/180.0
    u0 = r0*np.cos(sita0)
    v0 = r0*np.sin(sita0)
    # 軌跡を地図に表示
    plt.figure(figsize=(6, 6))
    m = Basemap(projection="merc", llcrnrlat=south, llcrnrlon=west, urcrnrlat=north, urcrnrlon=east, resolution="f")
    # 陸と海岸線と海
    m.drawcoastlines(color="k", linewidth=0.3)
    m.fillcontinents(color="y",lake_color="b")
    m.drawmapboundary()
    # 経線と緯線
    m.drawmeridians(np.arange(west, east, lonscale),labels=[0,0,0,1], linewidth=0.3, fontsize=9)
    m.drawparallels(np.arange(south, north, latscale),labels=[1,0,0,0], linewidth=0.3, fontsize=9)
    U, V, X, Y = m.rotate_vector(u0, v0, lons, lats, returnxy=True)
    # 矢印（ベクトル）
    cs1 = m.quiver(X, Y, U, V, speeds, norm=Normalize(vmin=0, vmax=40), edgecolor="k",  linewidth=.1, scale_units="inches", scale=1)
    # カラーバー
    bounds = np.array([0, 5, 10, 20, 30, 40, 50])
    cbar = m.colorbar(cs1,location="right", pad="10%", ticks=bounds)
    cbar.set_label("速力(ノット)")
    plt.suptitle("船の航跡 (経度，緯度，船首方位と速力)",fontsize=20, y=0.95)
    plt.title("航走ID ="+str(sailid), fontsize=16)
    plt.savefig("./tpm-ch12ex2-shiptrajectory-"+sailid+".png",dpi=360)
    plt.close()
    return 

# メイン関数
def main():
    sailsbysailid = readsailaisdata()
    sailid = "s000000"
    sailaisdata = sailsbysailid[sailid]
    drawtrajectory(sailid, sailaisdata)
    return

# ここから実行する
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("経過時間 =", elapsed_time)
    print("すべて完了 !!!")