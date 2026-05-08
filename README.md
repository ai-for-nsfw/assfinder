GHOST: AI ass detector for all type of asses (nude, non-nude, anime, NSFW, ...) 


## Description 

Model based on YOLO26 trained on a set of 600+ various ass images. Returns segmented output for images and video. Can be run on a single image, folder or images or video. 
Tested on python 3.12.5

## Example detection results

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
  pip install ultralytics opencv-python

	# for CPU inference:
  # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

  pip install -r requirements.txt
  ```

