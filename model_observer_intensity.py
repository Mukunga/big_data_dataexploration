import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
from collections import Counter

data_d = pd.read_csv("F:\\BG_DATA\\demongraphy.csv")
data_o = pd.read_csv("F:\\BG_DATA\\observations.csv")
data_b = pd.read_csv("F:\\BG_DATA\\block_copy.csv")
data_r = pd.read_csv("F:\\BG_DATA\\block_road_access_copy.csv")
data_l = pd.read_csv("F:\\BG_DATA\\block_landuse_copy.csv")
data_t = pd.read_csv("F:\\BG_DATA\\temperature_copy.csv")
data_p = pd.read_csv("F:\\BG_DATA\\precipitation_copy.csv")
# Remove the duplicates in temperature and precipitation
data_p = data_p.drop_duplicates(subset=["date_0","date_1","date_2","block"],keep="first") # Remove duplicated geometries
data_t = data_t.drop_duplicates(subset=["date_0","date_1","date_2","block"],keep="first")
# Merge data based on selected columns
data_br = pd.merge(data_b,data_r,how="left",on="block")
data_brl = pd.merge(data_br,data_l,how="left",on="block")
data_brl = data_brl.drop_duplicates(subset=["longit", "latit","geom"], keep="first")
data_brlt = pd.merge(data_brl,data_t,how="left",on="block")
data_brlt = data_brlt.dropna(subset=["temper"]) # Remove null value after merging with temperature
data_o["cnt"] = 1
#data_o = data_o.groupby(by = ["block"])["cnt"].agg({"cnt_sum":"sum"})
data_o = data_o.groupby(by = ["date_0","date_1","date_2","block"])["cnt"].agg({"cnt_sum":"sum"})  # Counting the number of observers in each block
#data_o.to_csv("F:\\big_data\\filter_data\\Obser.csv")
data_brlto = pd.merge(data_brlt,data_o,how="left",on=["date_0","date_1","date_2","block"])
data_brltop = pd.merge(data_brlto,data_p,how="left",on=["date_0","date_1","date_2","block"])
data_brltop = data_brltop.dropna(subset=["precip"]) # Remove null value after merging with precipitation
data_brltopd =  pd.merge(data_brltop,data_d,how="left",on="block")
data = data_brltopd.drop(["latit","longit","geom","scale_x","scale_y","id_x","lat_x","lon_x","corner_x","id_y","lat_y","lon_y","corner_y"],axis = 1)
# Delete some selected columns which will not be used in ML

# Road length
data_groad = data[data["maintainer"] == "G"] # Extract the road data which is assigned to 'G'
data_groad["G_ROADLENGTH"] = data_groad["roadlength"]
data_groad["P_ROADLENGTH"] = 0

data_proad = data[data["maintainer"] == "P"]
data_proad["G_ROADLENGTH"] = 0
data_proad["P_ROADLENGTH"] = data_proad["roadlength"]
data_oroad = data[data["maintainer"] != "G"]
data_oroad = data_oroad[data_oroad["maintainer"] != "P"]
data_oroad["G_ROADLENGTH"] = 0
data_oroad["P_ROADLENGTH"] = 0

data_road = data_oroad.append(data_proad)
data_road = data_road.append(data_groad)
data_road = data_road.drop(["maintainer","roadlength"],axis=1)


# The classification of landuse
data_bos = data_road[data_road["category"] == "Bos"]
data_bos["BOS_AREA"] = data_bos["areasum"]
data_bos["BEBOUWD_AREA"] = 0
data_bos["BEDRIJFSTERREIN_AREA"] = 0
data_bos["DROOG_AREA"] = 0
data_bos["GLAS_AREA"] = 0
data_bos["HOOFDWEG_AREA"] = 0
data_bos["LANDBOUW_AREA"] = 0

data_Bebouwd = data_road[data_road["category"] == "Bebouwd"]
data_Bebouwd["BOS_AREA"]=0
data_Bebouwd["BEBOUWD_AREA"] = data_Bebouwd["areasum"]
data_Bebouwd["BEDRIJFSTERREIN_AREA"] = 0
data_Bebouwd["DROOG_AREA"] = 0
data_Bebouwd["GLAS_AREA"] = 0
data_Bebouwd["HOOFDWEG_AREA"] = 0
data_Bebouwd["LANDBOUW_AREA"] = 0

data_Bedrijfsterrein = data_road[data_road["category"]=="Bedrijfsterrein"]
data_Bedrijfsterrein["BOS_AREA"] = 0
data_Bedrijfsterrein["BEBOUWD_AREA"] = 0
data_Bedrijfsterrein["BEDRIJFSTERREIN_AREA"] = data_Bedrijfsterrein["areasum"]
data_Bedrijfsterrein["DROOG_AREA"] = 0
data_Bedrijfsterrein["GLAS_AREA"] = 0
data_Bedrijfsterrein["HOOFDWEG_AREA"] = 0
data_Bedrijfsterrein["LANDBOUW_AREA"] =0

data_Droog = data_road[data_road["category"]=="Droog natuurlijk terrein"]
data_Droog["BOS_AREA"] = 0
data_Droog["BEBOUWD_AREA"] = 0
data_Droog["BEDRIJFSTERREIN_AREA"] = 0
data_Droog["DROOG_AREA"] = data_Droog["areasum"]
data_Droog["GLAS_AREA"] = 0
data_Droog["HOOFDWEG_AREA"] = 0
data_Droog["LANDBOUW_AREA"] = 0

data_Glastuinbouw = data_road[data_road["category"]=="Glastuinbouw"]
data_Glastuinbouw["BOS_AREA"] = 0
data_Glastuinbouw["BEBOUWD_AREA"] = 0
data_Glastuinbouw["BEDRIJFSTERREIN_AREA"] = 0
data_Glastuinbouw["DROOG_AREA"] = 0
data_Glastuinbouw["GLAS_AREA"] = data_Glastuinbouw["areasum"]
data_Glastuinbouw["HOOFDWEG_AREA"] = 0
data_Glastuinbouw["LANDBOUW_AREA"] = 0

data_Hoofdweg = data_road[data_road["category"]=="Hoofdweg"]
data_Hoofdweg["BOS_AREA"] = 0
data_Hoofdweg["BEBOUWD_AREA"] = 0
data_Hoofdweg["BEDRIJFSTERREIN_AREA"] = 0
data_Hoofdweg["DROOG_AREA"] = 0
data_Hoofdweg["GLAS_AREA"] = 0
data_Hoofdweg["HOOFDWEG_AREA"] = data_Hoofdweg["areasum"]
data_Hoofdweg["LANDBOUW_AREA"] = 0

data_Landbouw = data_road[data_road["category"]=="Landbouw"]
data_Landbouw["BOS_AREA"] = 0
data_Landbouw["BEBOUWD_AREA"] = 0
data_Landbouw["BEDRIJFSTERREIN_AREA"] = 0
data_Landbouw["DROOG_AREA"] = 0
data_Landbouw["GLAS_AREA"] = 0
data_Landbouw["HOOFDWEG_AREA"] = 0
data_Landbouw["LANDBOUW_AREA"] = data_Landbouw["areasum"]

data_empty = data_road[data_road["category"]!="Landbouw"]
data_empty = data_road[data_road["category"]!="Bos"]
data_empty = data_road[data_road["category"]!="Bebouwd"]
data_empty = data_road[data_road["category"]!="Bedrijfsterrein"]
data_empty = data_road[data_road["category"]!="Droog natuurlijk terrein"]
data_empty = data_road[data_road["category"]!="Glastuinbouw"]
data_empty = data_road[data_road["category"]!="Hoofdweg"]
data_empty["BOS_AREA"] = 0
data_empty["BEBOUWD_AREA"] = 0
data_empty["BEDRIJFSTERREIN_AREA"] = 0
data_empty["DROOG_AREA"] = 0
data_empty["GLAS_AREA"] = 0
data_empty["HOOFDWEG_AREA"] = 0
data_empty["LANDBOUW_AREA"] = 0
data_empty["LANDBOUW_AREA"] = 0

data_category = data_bos.append(data_Bebouwd)
data_category = data_category.append(data_Bedrijfsterrein)
data_category = data_category.append(data_Droog)
data_category = data_category.append(data_Glastuinbouw)
data_category = data_category.append(data_Hoofdweg)
data_category = data_category.append(data_Landbouw)
data_category = data_category.append(data_empty)
data_category = data_category.drop(["category","areasum"],axis = 1)

# Clustering of observer intensity
data_category["other"] = 0
data_category = data_category.fillna(0)
attribution = np.array(data_category[["cnt_sum","other"]])
category = KMeans(n_clusters=5)
k_m = category.fit(attribution)
data_category['label']=k_m.labels_

# DT
print(Counter(data_category["label"]))
data_lable0 = data_category[data_category["label"]==0]
data_lable1 = data_category[data_category["label"]!=0]
a = random.sample(range(0,470539),500)
data_lable0 = data_lable0.iloc[a,:]
data_category = data_lable0.append(data_lable1)
data_category.info()
#data_category.to_csv("C:\\Users\\Administrator\\Desktop\\r.csv")
data_DT = data_category.drop(["date_0","date_1","date_2","block","cnt_sum","label","other"],axis = 1)
DT_result = data_category["label"]
x_train, x_test, y_train, y_test = train_test_split(data_DT,DT_result,test_size = 0.25,random_state = 7)
print(Counter(y_train))
DT = tree.DecisionTreeClassifier(max_depth=7)
DT.fit(x_train, y_train)


from sklearn import tree
import pydotplus
x = pd.DataFrame(x_train)
y = pd.DataFrame(y_train)
dot_data = tree.export_graphviz(DT, out_file=None,feature_names=x.columns,class_names=["lowest","lower","middle","higher","highest"])
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_jpg("tree.jpg")
graph.write_jpg("F:\\big_data\\filter_data\\tree.jpg")
result = DT.predict(x_test)
Roc = pd.DataFrame(result, y_test)
Roc.to_csv("F:\\big_data\\filter_data\\Roc.csv")
print(accuracy_score(y_test, DT.predict(x_test)))
