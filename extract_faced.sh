#!/bin/bash
cd /home/ubuntu/face-detection && python extract_faced.py --dataset dataset \
	--embeddings output/embeddings_faced.pickle \
	--detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7
