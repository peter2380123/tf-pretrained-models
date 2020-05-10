"""

Run this script after starting 'start-docker'.

This python script evaluates the input image and outputs textual result.
Does not draw any bounding box nor displaying any image. 

"""

import PIL.Image
import numpy
import requests
from pprint import pprint
import time

image = PIL.Image.open("../../images/dog.jpg")  # Change dog.jpg with your image
image_np = numpy.array(image)


payload = {"instances": [image_np.tolist()]}
start = time.perf_counter()
res = requests.post("http://localhost:8080/v1/models/default:predict", json=payload)
print(f"Took {time.perf_counter()-start:.2f}s")
#pprint(res.json())

# Retrieve data from response, cleanly with zeros excluded.
box_data = res.json().get('predictions', {})[0].get('detection_boxes')
box_label = res.json().get('predictions', {})[0].get('detection_classes')
box_score = res.json().get('predictions', {})[0].get('detection_scores')
valid_detect_count = int(res.json().get('predictions', {})[0].get('num_detections'))

# Get width and height of image for later use
width, height = image.size

result_dict = []

for count,box in enumerate(box_data, 0): # should exclude invalid boxes
  # discontinue after valid boxes are processed
  if(count >= valid_detect_count):
    break
  
  #y1, x1, y2, x2 = box[0], box[1], box[2], box[3] # get x y coords for each box
  y1, x1, y2, x2 = (int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)) # get x y coords for each box

  
  new_dict = {"image_id":0,"category_id":count,"bbox":[y1, x1, y2-y1, x2-x1],"score":box_score[count]}
  
  result_dict.append(new_dict)

print(f"Valid boxes: {valid_detect_count:d}")
print(result_dict)