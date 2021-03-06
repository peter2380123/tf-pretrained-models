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
dir_list = natsorted(os.listdir(img_dir))

start = time.perf_counter()

#for image_id, img_path in enumerate(dir_list, start=1):
  
  #if(image_id >= 10):
    #break
  #print(f"image_id: {image_id:d}") # print image id as sanity check
  
def process_image(name):
  result_dict = []

  input_path = os.path.join(img_dir, name)

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

    new_dict = {"image_id":int(name.split('.')[0]),"category_id":box_label[count],"bbox":[y1, x1, y2-y1, x2-x1],"score":box_score[count]}
    
    result_dict.append(new_dict)


  if len(result_dict) is not 0:
    print(result_dict) # print result as sanity check 
    return result_dict

pool = Pool()
result = [x for x in pool.map(process_image, dir_list) if x is not None for x in x]

pool.close()
pool.join()

print(f"Whole process took {time.perf_counter()-start:.2f}s")

print(result)

# write result to json
with open("results/multiprocess-10k-imgs.json", 'w') as outfile:
  json.dump(result, outfile)