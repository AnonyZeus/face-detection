# USAGE
# python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle \
#	--detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

from faced import FaceDetector
from faced.utils import annotate_image

from config_reader import read_config


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=None, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind('.'):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath

file_paths, configs = read_config()

# load our serialized face detector from disk
print('[INFO] loading face detector...')
face_detector = FaceDetector()

# load our serialized face embedding model from disk
print('[INFO] loading embedder from {}'.format(file_paths['embedder_path']))
embedder = cv2.dnn.readNetFromTorch(file_paths['embedder_path'])

# grab the paths to the input images in our dataset
print('[INFO] quantifying faces...')
current_dir = os.path.dirname(os.path.realpath(__file__))
imagePaths = list(list_images(os.path.join(current_dir, 'dataset')))

# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []
# glb_confidence = 0.5

# initialize the total number of faces processed
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print('[INFO] processing image {}/{}'.format(i + 1,
                                                    len(imagePaths)))
    print('[INFO] image: {}'.format(imagePath))
    name = imagePath.split(os.path.sep)[-2]

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    image = cv2.imread(imagePath)
    # image = imutils.resize(image, width=600)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = face_detector.predict(image, float(configs['confidence']))

    # ensure at least one face was found
    print('[INFO] number of detected faces: {}'.format(len(bboxes)))
    if len(bboxes) > 0:
        for xb, yb, wb, hb, pb in bboxes:
            startX = int(xb - wb/2)
            startY = int(yb - hb/2)
            endX = int(xb + wb/2)
            endY = int(yb + hb/2)

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            # (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            # if fW < 20 or fH < 20:
            #     continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # add the name of the person + corresponding face
            # embedding to their respective lists
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

# dump the facial embeddings + names to disk
print('[INFO] serializing {} encodings...'.format(total))
data = {'embeddings': knownEmbeddings, 'names': knownNames}
f = open(os.path.join(current_dir, 'output', 'embeddings_faced.pickle'), 'wb')
f.write(pickle.dumps(data))
f.close()