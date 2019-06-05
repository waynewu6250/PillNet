# !brew install poppler
# !pip install pdf2image

import pandas as pd
import numpy as np
import cv2
import os
import re
import requests
from pdf2image import convert_from_path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def scraping(df):

    counter = 1
    for name, url in zip(df["英文品名"].values, df["外觀圖檔連結"].values):
        
        if type(url) != str or type(name) != str:
            continue
        
        name = re.sub(r"[\"\/\.]", "", name)
        filepdf = "images/{}.pdf".format(name)
        fileimg = "images/{}.jpg".format(name)
        
        if os.path.exists(fileimg):
            print('{}: {} file exists!'.format(counter, name))
        else:
            try:
                r = requests.get(url, stream=True, timeout=20)
                r.raise_for_status()
                with open(filepdf, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                pages = convert_from_path(filepdf, 500)
                for page in pages:
                    page.save(fileimg, 'JPEG')
                print("{}: {} image download".format(counter, name))
                os.remove(filepdf)

            except:
                pass
        
        counter += 1

def resize(basewidth):
    
    for i, file in enumerate(os.listdir('./images/')):
        filename = os.path.join('./images/',file)
        img = cv2.imread(filename)
        try:
            if (img.shape[0]*img.shape[1]) >= 178956970:
                os.remove(filename)
                continue
        except:
            continue
        img = Image.open(filename)
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        img.save(os.path.join(filename))
        
        print("{}: Done with reshape {}".format(i,file))

if __name__ == "__main__":
    
    df = pd.read_excel("raw_data.xlsx")
    df["英文品名"] = df["英文品名"].map(lambda x: str(x).replace(" ", "_"))

    # scraping
    scraping(df)

    # Select the imgs of tablets or capsules
    for file in os.listdir("./images/"):
        newname = os.path.join("./images/",file.replace('“', '_').replace('”', '_'))
        os.rename(os.path.join("./images/",file), newname)
        find1 = file.lower().find("tablets")
        find2 = file.lower().find("capsules")
        if find1*find2 == 1:
            os.remove(newname)
    
    # Resize all the images with proper size
    resize(500)

    