import sys
"""You need to have the Tensorflow Object Detection API installed for this code to work
This file needs to be stored in the Object Detection folder of Object Detection API
"""
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import cv2
import winsound

sys.path.append("..")
cap = cv2.VideoCapture("Test_Footage.mp4")
#fps = 30
#capSize = (640, 480) 
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter()
#success = out.open('output_video.avi',30,fps,fourcc,capSize,True)

# local path!
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'faster_rcnn_inception_v2_coco_2018_01_28, output_inference_graph_v6.pb faster_rcnn_inception_v2_pets
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = 'C:/Car_detection/output_inference_graph_v5.pb ssdlite_mobilenet/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('C:/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt')

firstDetectWarn = False
NUM_CLASSES = 5

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())

frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while cap.isOpened():
      ret, image_np = cap.read()
      if ret == True:
          t1 = cv2.getTickCount()
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=2)
          cv2.putText(image_np,"FPS: {0:.2f}".format(frame_rate_calc),(550,20),font,0.5,(255,255,0),2,cv2.LINE_AA)
          t2 = cv2.getTickCount()
          time1 = (t2-t1)/freq
          frame_rate_calc = 1/time1
        
      for i,b in enumerate(boxes[0]):
        
         if classes[0][i] == 1: #STOP SIGN DETECTION
          if scores[0][i] >= 0.5:
            cv2.putText(image_np, 'STOP!!!', (300,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

         if classes[0][i] == 2 or classes[0][i] == 3 or classes[0][i] == 4:#Distances

          if scores[0][i] >= 0.5:
            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
            apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*640),int(mid_y*480)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


            if apx_distance <=0.3 and apx_distance >0:
              if mid_x > 0.35 and mid_x < 0.65:
                if firstDetectWarn == False:
                 cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                 print(firstDetectWarn,'1')
                 winsound.PlaySound("beep.wav", winsound.SND_ASYNC)
                 
              else:
                 firstDetectWarn = True
                 print(firstDetectWarn,'2')
          cv2.imshow('Output',cv2.resize(image_np,(640,480)))
      #reset  
      if firstDetectWarn == True:
         firstDetectWarn = False



      if cv2.waitKey(25) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              break

cap.release()
#out.release()
cv2.destroyAllWindows()
