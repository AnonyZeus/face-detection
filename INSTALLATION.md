# Installation
## Create virtual environment
 > conda create -n face python=3
## Install necessary libraries
 > conda activate face

 > conda install -c menpo opencv3

 > conda install scikit-learn

 > pip install -r requirement.txt

This step can be ognored because an environment "face" was already created on server

# Deploy Server
Steps to deploy face detection server
 > cd <path_to_face_detector_folder>

 > ./start_server.sh

# Manually run auto collect-training data
> cd <path_to_face_detector_folder>

 > ./auto_sync.sh

# Userful APIs
## Setup stream url
 Make a post request to <server_id>:5000/change_stream_url with body is a json string.

 Body example
 >{
	"stream_url": "rtsp://user:password@182.171.230.167:10554/ipcam_mjpeg.sdp",
	"auth_id": "auth_info",
	"monitor_id": <id_of_monitor>
 }

## Validate an image
 Make a post request to <server_id>:5000/valid_image with body type is formdata, and have "file" key that point to image to be validated.

## Receive realtime face detection result in json format
 Make a get request to <server_id>:5000/view_json

## Record video from remote stream

## Get streaming output