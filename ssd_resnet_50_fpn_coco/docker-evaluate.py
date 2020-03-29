import PIL.Image
import numpy
import requests
from pprint import pprint
import time
from IPython.display import display
import cv2
import matplotlib.pyplot as plt
import json
import sys

image = PIL.Image.open("../../images/dog_cat.jpg")  # Change dog.jpg with your image
image_np = numpy.array(image)


payload = {"instances": [image_np.tolist()]}
start = time.perf_counter()
res = requests.post("http://localhost:8080/v1/models/default:predict", json=payload)
print(f"Took {time.perf_counter()-start:.2f}s")
#display(PIL.Image.fromarray(image_np))

# Retrive data from response
box_data = res.json().get('predictions', {})[0].get('detection_boxes')
box_label = res.json().get('predictions', {})[0].get('detection_classes')
box_score = res.json().get('predictions', {})[0].get('detection_scores')

# Declare classes with external txt file
class_dict = {}
with open("coco-label.txt") as f:
  line = f.readline()
  cnt = 1
  while line:
    #print("Line {}: {}".format(cnt, line.strip()))
    class_dict[cnt] = line.strip()
    line = f.readline()
    cnt += 1

# Get width and height of image for later use
width, height = image.size

# Create a colour palette for later use
colour_list = []
with open('colour_list.json') as json_file:
  data = json.load(json_file)
  for e in data:
    rgb_dict = e.get('rgb', {})
    rgb_list = [rgb_dict.get('r'), rgb_dict.get('g'), rgb_dict.get('b')]
    colour_list.append(tuple(rgb_list))

# Draw detection boxes along with labels 
def drawBoxes(img, data):
  box_index = 0 # keeps track of boxes
  colour_index = 0 # for colour palette

  for box in data:
    y1, x1, y2, x2 = (int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)) # get x y coords for each box
    
    # Exclude boxes of 0's
    if(y1==0 and x1==0 and y2==0 and x2==0):
      continue

    print(box) # sanity check, should see no 0-only boxes

    # Loop back colour index if more boxes than palette
    if(colour_index > len(colour_list)-1):
      colour_index = 0
    
    cv2.rectangle(img, (x1,y1), (x2,y2), colour_list[colour_index], 3)
    label = class_dict.get(box_label[box_index]) # get label from earlier created dictionary
    score = box_score[box_index]
    label = label + " %.3f"%score
    print(label) # report for label type
    cv2.putText(img, label, (x1, y1 - 12), 0, 1.7e-3 * height, colour_list[colour_index], 2) # writes label to image

    colour_index += 1
    box_index += 1

  return img

cv_image = numpy.array(image)
boxed_img = drawBoxes(cv_image, box_data)
plt.imshow(boxed_img, aspect="auto")
plt.show()
