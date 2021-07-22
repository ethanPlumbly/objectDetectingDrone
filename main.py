import numpy as np
import math
import time
import tensorflow as tf
import cv2
from djitellopy import Tello
import time
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def run_inference_for_single_image(model, image):
    # Image formatting to Tensor
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference and extract outputs in a readable format
    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)    
    return output_dict


def run_inference(model, category_index, cap):
    global frameCount
    frameCount = 0
    while True:
        # Retrieve current frame
        image_np = cap.frame

        # Run inference on single image
        output_dict = run_inference_for_single_image(model, image_np)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        boxes = output_dict['detection_boxes']

        # Extract box coordinates
        ymin = boxes[0][0]*height
        xmin = boxes[0][1]*width
        ymax = boxes[0][2]*height
        xmax = boxes[0][3]*width

        # Calculate box parameters for autonomous control
        boxWidth = xmax-xmin
        boxHeight = ymax-ymin
        boxX = (xmax+xmin)*0.5
        boxY = (ymax+ymin)*0.5
        boxArea = boxWidth*boxHeight

        # Visualise capture footage with bounding box overlay
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))

        # Close capture window when d key is pressed
        if cv2.waitKey(20) & 0xFF == ord('d'):
            cap.release()
            cv2.destroyAllWindows()
            break

        # Calculate orthogonal velocities using box attributes
        xSpeed = (boxX - width / 2) / 20
        ySpeed = (height / 2 - boxY) / 20
        zSpeed = (450 - math.sqrt(boxArea)) / 10

        # Send calculated orthogonal velocities to drone
        drone.send_rc_control(xSpeed, zSpeed, ySpeed, 0)
        frameCount += 1

# Connect to Tello drone using djitellopy
drone = Tello()
drone.connect()
drone.streamoff() 
drone.streamon()

# Footage parameters
width = 2592
height = 1936

# Load model and label map
model = load_model('mobilenet')
category_index = label_map_util.create_category_index_from_labelmap('label_map.pbtxt', use_display_name=True)

# Retreive drone capture
capture = drone.get_frame_read()

# Log current time and run inference, once inference calculate log time again to calculate fps
t1 = time.time()
run_inference(model, category_index, capture)
t2 = time.time()
print("fps: ",frameCount/(t2 - t1))