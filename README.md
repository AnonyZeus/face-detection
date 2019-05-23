# Steps to start detection system
After executed these steps, user can start using face detection function on aimcloud

## Start face detection server

> ssh ubuntu@13.231.21.118

> conda activate face

> cd face-detector

> ./start_server.sh

## Start auto analyzing daemon

> ssh ubuntu@54.238.198.114

> python image_detection_daemon.py
