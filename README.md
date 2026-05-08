GHOST: AI ass detector for all type of asses (nude, non-nude, anime, NSFW, ...) 


## Description 

Model based on YOLO26 trained on a set of 600+ various ass images. Returns segmented output for images and video. Can be run on a single image, folder or images or video. 

## Example detection result

<div>
<img src="/examples/DominikaSvanovaAssLabeled.webp" width="360"/>
<div/>

## Installation
  
1. Clone this repository
  ```bash
  git clone https://github.com/ai-for-nsfw/assfinder
  cd assfinder
  ```
2. Install dependent packages
  ```bash

  pip install -r requirements.txt
  ```
  If it is not possible to install onnxruntime-gpu, try onnxruntime instead  
  