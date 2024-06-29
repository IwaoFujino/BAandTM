# Basemapで関東地方の地図を作成
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import japanize_matplotlib

# 関東地方
south = 35.0 # 緯度（南（下））
north = 36.5 # 緯度（北（上））
west = 139.0 # 経度（西（左））
east = 141.0 # 経度（東（右））
lonscale = 0.5
latscale = 0.5
# 地図を作成
plt.figure(figsize=(8, 8))
map = Basemap(projection="merc", llcrnrlat=south, urcrnrlat=north, llcrnrlon=west, urcrnrlon=east, resolution="f")
map.drawcoastlines(color="k", linewidth=0.2)
map.fillcontinents(color="lightgreen",lake_color="b")
map.drawmapboundary(fill_color="aqua")
map.readshapefile("./gadm36_JPN_shp/gadm36_JPN_1", "prefectural_bound1", color="k", linewidth=.3) # 県境
map.readshapefile("./gadm36_JPN_shp/gadm36_JPN_2", "prefectural_bound2", color="m", linewidth=.2) # 市町村境
map.drawmeridians(np.arange(west,east+lonscale,lonscale),labels=[0,0,1,1],fontsize=12)
map.drawparallels(np.arange(south,north+latscale,latscale),labels=[1,1,0,0],fontsize=12)
mapfile = "tpm-ch12ex1-fig1.png"
plt.savefig(mapfile)