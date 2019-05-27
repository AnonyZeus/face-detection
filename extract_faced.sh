#!/bin/bash
cd /home/ubuntu/face-detection && python extract_data.py --dataset dataset \
	--embeddings output/embeddings_faced.pickle \
	--detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7

cd /home/ubuntu/face-detection && ./extract_faced.sh
cd /home/ubuntu/face-detection && ./train_faced.sh
