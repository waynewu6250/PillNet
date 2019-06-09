# PillNet
PillNet: Pharmaceutical Pill identification via Tensorflow Object Detection API and Inception-ResNet CNN-fused siamese network.

## Structure
The project has been seperated into three parts:
1. Object detection
2. Pill recognition
3. App platform

* **Folder Structure**
```
align/ (all efforts on object detection)
└── data/
    ├── pill_labels.csv
    ├── train_labels.csv
    ├── test_labels.csv
    ├── train.record
    ├── test.record

└── raw_data/
    └── annotations/
        ...
    └── images/
        ...
    ├── pill_download.py
    ├── raw_data.xlsx

└── training/
    ├── pill_detection.pbtxt
    ├── ssd_mobilenet_v1_pill.config
    ├── train.sh
    ├── export.sh

└── utils/
    ├── dataset_util.py
    ├── label_map_util.py
    ├── visualization_utils.py

└── my_exported_graphs/
    (checkpoints trained from api)

├── generate_tfrecord.py
├── xml_to_csv.py

-----------------------------------

data/ (cropped images for object recognition)
    └── train_imgs/
        ...
    ├── data.py

-----------------------------------

model/
    ├── inception_resnet_v1.py
    ├── model.py
checkpoints/
graph/

-----------------------------------

train.py
livestream.py
config.py
pill_crop.py


```

## To USE

### Object Detection
1. Download all the pill images using `align/raw_data/raw_data.xlsx` by using the following command:

>
    python align/raw_data/pill_download.py

2. Use [LabelImg](https://github.com/tzutalin/labelImg) to label the data for tensorflow object detection api input. And store the xml output files all in  `align/raw_data/annotations/`

>
    pip install LabelImg
    labelImg

3. Then use the following command to transform the data into TFrecord format to feed into network.

>
    python align/xml_to_csv.py

    python align/generate_tfrecord.py \
            --csv_input=align/data/train_labels.csv \
            --output_path=align/data/train.record \
            --image_dir=align/raw_data/images \
    
    python align/generate_tfrecord.py \
            --csv_input=align/data/test_labels.csv \
            --output_path=align/data/test.record \
            --image_dir=align/raw_data/images \

4. Put the required files into tensorflow object detection modules.

>
    git clone https://github.com/tensorflow/models.git

Then, locate into `models/research/object_detection/` folder. And put all the following files into `data` folder. Also go to [Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) to download the required models. Here we use ssd_mobilenet_v1_coco, download and unzip it into object_detection folder too.
```
object_detection/
    └── data/
        ├── pill_detection.pbtxt
        ├── ssd_mobilenet_v1_pill.config
        ├── train.record
        ├── test.record

    └── training/

    └── my_exported_graphs/

    ├── train.sh
    ├── export.sh
    ├── livestream.py

```

5. Then use the following command to train the module.
>
    bash train.sh

And export your model by typing the following command:
>
    bash export.sh <Your checkpoint id>

Your trained model will be saved in `object_detection/my_exported_graphs/`

6. Move your `my_exported_graphs` back to PillNet folder and put it down in `PillNet/align/` folder. Then you can run three modes to validate your model:
>
    1. Real-time
    python livestream.py -m livestream

    2. One test image
    python livestream.py -m test -i <Test Image Path>

    3. Test images
    python livestream.py -m images


### Pill Recognition
1. First run the following command to create training data for pills
>
    python pill_crop.py



    