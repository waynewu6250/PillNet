import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
from PIL import Image
import pandas as pd

from config import opt
from utils import *
from align.utils import label_map_util
from align.utils import visualization_utils as vis_util

# Detection helper
def detect(image_np, detection_modules):

    # Phase I: Object detection
    (image_tensor, detection_boxes, detection_scores,
     detection_classes, num_detections) = detection_modules

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores,
                    detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

    # Phase II: Pill Recognition
    image_np_recognized = image_np.copy()
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)
    with open(opt.label_save_path, "rb") as f:
        label_ref = pickle.load(f)

    classes_recognized, scores_recognized = recognize_pill(image_np, boxes, classes, scores, label_ref["labels"])
    category_index_recognized = {k:{"name":v} for k,v in label_ref["id2class"].items()}

    # Draw the detection result
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4)
    
    # Draw the recognition result
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np_recognized,
        boxes,
        classes_recognized,
        scores_recognized,
        category_index_recognized,
        use_normalized_coordinates=True,
        line_thickness=4)
    
    return image_np, image_np_recognized

# Recognize image
def recognize_image(detection_modules, image_path):

    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)

    # Detect the image
    image_np, image_np_recognized = detect(image_np, detection_modules)

    save_path = "./data/identify_results/{}".format(image_path.split('/')[-1].split('.')[0])
    
    cv2.imwrite(save_path+'-result.jpg', image_np)
    cv2.imwrite(save_path+'-recognized.jpg', image_np_recognized)
    
    return image_np, image_np_recognized

# Live Stream Mode
def start_livestream(detection_modules):

    # Start to detect
    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):  # check !
        # capture frame-by-frame
        ret, frame = cap.read()

        if ret:  # check ! (some webcam's need a "warmup")
            
            # Detect the frame
            frame, frame_recognized = detect(frame, detection_modules)

            # Display the resulting frame
            cv2.imshow('pill detection', frame_recognized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="which mode to perform: livestream or image", dest="mode", default="livestream")
    parser.add_argument("-i", "--image", help="The image path you would like to test", dest="img_path", default="./align/raw_data/tests/YuLuAn_Cold_FC_Tablets.jpg")
    args = parser.parse_args()

    # Import the object detection graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(opt.PATH_TO_CKPT, 'rb') as fid:
            od_graph_def.ParseFromString(fid.read())
            tf.import_graph_def(od_graph_def, name='')

    # Load Label map
    label_map = label_map_util.load_labelmap(opt.PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=opt.NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            image_tensor = detection_graph.get_tensor_by_name(
                'image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            detection_modules = (image_tensor, detection_boxes,
                                 detection_scores, detection_classes,
                                 num_detections)
            # Live Stream Mode
            if args.mode == "livestream":
                start_livestream(detection_modules)
            
            # Test single image
            elif args.mode == "test":
                opt.image_path = args.img_path
                image_np, image_np_recognized = recognize_image(detection_modules, opt.image_path)
                cv2.imshow("Label Img",image_np)
                cv2.imshow("Label Img Recognized",image_np_recognized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Validation
            elif args.mode == "images":
                test_df = pd.read_csv("align/data/test_labels.csv")
                for i, filename in enumerate(test_df["filename"]):
                    image_path = os.path.join("align/raw_data/tests", filename)
                    image_np, image_np_recognized = recognize_image(detection_modules, image_path)
                    print("Identification Done: %d" % i)
            else:
                start_livestream(detection_modules)
