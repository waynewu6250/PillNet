import pandas as pd
import cv2
from PIL import Image
import os

data = pd.read_csv("./align/data/pill_labels.csv")

for i in range(len(data)):
    
    filename = data.loc[i]["filename"]
    xmin = data.loc[i]["xmin"]
    ymin = data.loc[i]["ymin"]
    xmax = data.loc[i]["xmax"]
    ymax = data.loc[i]["ymax"]
    counter = data.loc[i]["data_num"]
    
    path = os.path.join("./align/raw_data/tests",filename)
    if os.path.exists(path):
        im = Image.open(path)
        region = im.crop((xmin, ymin, xmax, ymax))
        region.save("./data/train_imgs/"+filename.split('.')[0]+"-{}.jpg".format(counter))