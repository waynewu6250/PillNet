import tensorflow as tf
import numpy as np
import cv2
from config import opt

from align.utils import label_map_util
from align.utils import visualization_utils as vis_util

# Import the graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(opt.PATH_TO_CKPT, 'rb') as fid:
        od_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(od_graph_def, '')

# Load Label map
label_map = label_map_util.load_labelmap(opt.PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=opt.NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# helper
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Start to detect
cap = cv2.VideoCapture(0)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

        while(cap.isOpened()):  # check !
            # capture frame-by-frame
            ret, frame = cap.read()

            if ret:  # check ! (some webcam's need a "warmup")
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
                image_np_expanded = np.expand_dims(frame, axis=0)

                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4)

                # Display the resulting frame
                cv2.imshow('pill detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything is done release the capture
        cap.release()
        cv2.destroyAllWindows()
