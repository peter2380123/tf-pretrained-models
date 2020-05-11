"""

Run this script after starting 'start-docker'.

This python script evaluates the input image and outputs textual result.
Does not draw any bounding box nor displaying any image. 

Also tries to process multiple images in one execution.

"""

import PIL.Image
import numpy
import requests
from pprint import pprint
import time
import os
import json
from natsort import natsorted
from multiprocessing import Pool

img_dir = "../../driving-in-the-matrix-images/repro-10k-images/VOC2012/JPEGImages/"
result_dict = []
dir_list = natsorted(os.listdir(img_dir))

start = time.perf_counter()

for image_id, img_path in enumerate(dir_list, start=1): # start ID'ing from 1 to comply with file names
  if(image_id >= 10):
    break
  print(f"image_id: {image_id:d}") # print image id as sanity check

  input_path = os.path.join(img_dir, img_path)

  image = PIL.Image.open(input_path)  # Change dog.jpg with your image
  image_np = numpy.array(image)


  payload = {"instances": [image_np.tolist()]}
  #start = time.perf_counter()
  res = requests.post("http://localhost:8080/v1/models/default:predict", json=payload)
  #print(f"Took {time.perf_counter()-start:.2f}s")
  #pprint(res.json())

  # Retrieve data from response, cleanly with zeros excluded.
  box_data = res.json().get('predictions', {})[0].get('detection_boxes')
  box_label = res.json().get('predictions', {})[0].get('detection_classes')
  box_score = res.json().get('predictions', {})[0].get('detection_scores')
  valid_detect_count = int(res.json().get('predictions', {})[0].get('num_detections'))

  # Get width and height of image for later use
  width, height = image.size

  

  for count,box in enumerate(box_data, start=0): # should exclude invalid boxes
    # discontinue after valid boxes are processed
    if(count >= valid_detect_count):
      break
    
    #y1, x1, y2, x2 = box[0], box[1], box[2], box[3] # get x y coords for each box
    y1, x1, y2, x2 = (int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)) # get x y coords for each box

    
    new_dict = {"image_id":image_id,"category_id":box_label[count],"bbox":[y1, x1, y2-y1, x2-x1],"score":box_score[count]}
    
    result_dict.append(new_dict)

print(result_dict) # print result as sanity check 

print(f"Whole process took {time.perf_counter()-start:.2f}s")

#import sys
#sys.exit()

# write result to json
with open("results/test-slow-10-imgs.json", 'w') as outfile:
  json.dump(result_dict, outfile)