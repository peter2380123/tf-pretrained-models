## Installation
For complete steps, see [Tensorflow's Object Detection Installation Guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

No additional package installation required (not even Tensorflow, or at least so far). 

## Quick Start
Run the 'start-docker' shell script and let it do magic. 

## What's going on 
A docker script (see [this link](https://medium.com/@pierrepaci/deploy-tensorflow-object-detection-model-in-less-than-5-minutes-604e6bb0bb04)) runs Tensorflow Serving API which allows quick loading of Tensorflow's pretrained models. 

A custom evaluation python script is then written by me, which draws the detection boxes and relative scores. 
