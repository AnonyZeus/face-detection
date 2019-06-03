#!/bin/bash
cd /home/ubuntu/face-detection/
python auto_sync.py -p /home/ubuntu/aim-cloud/application/webroot/files/user_images_list.txt -i ubuntu@54.238.198.114 -d /home/ubuntu/face-detection/dataset -f true
# cd /home/ubuntu/face-detection && cp -r 0 dataset/
cd /home/ubuntu/face-detection && ./extract_faced.sh
cd /home/ubuntu/face-detection && ./train_faced.sh
kill -9 $(pgrep -f main.py)
sh start_server.sh
