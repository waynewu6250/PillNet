import os
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def xml_to_csv(path):

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    
    return xml_df

def split_labels(df):
    
    df["data_num"] = np.arange(0,len(df))
    gb = df.groupby('data_num')
    dfs = [gb.get_group(x) for x in gb.groups]
    indexes = np.random.permutation(len(df))
    
    train = pd.concat(dfs[i] for i in indexes[:int(len(df)*0.8)])
    test = pd.concat(dfs[i] for i in indexes[int(len(df)*0.8):])
    
    return train, test


if __name__ == "__main__":
    image_path = os.path.join(os.getcwd(), 'raw_data/annotations')
    xml_df = xml_to_csv(image_path)
    train, test = split_labels(xml_df)
    xml_df.to_csv('data/pill_labels.csv', index=None)
    train.to_csv('data/train_labels.csv', index=None)
    test.to_csv('data/test_labels.csv', index=None)
    print('Successfully converted xml to csv.')