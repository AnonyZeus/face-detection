#!/bin/bash
python train_models.py --embeddings output/embeddings_faced.pickle \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
