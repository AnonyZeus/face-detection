import os
import imutils
import pickle
import time
import cv2
import threading
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import json
import datetime
import requests

ZM_URL = 'http://18.179.207.49/zm'
ZM_STREAM_URL = f'{ZM_URL}/cgi-bin/nph-zms'
LOGIN_URL = f'{ZM_URL}/api/host/login.json?user=admin&pass=admin'



class Camera(object):
    thread_list = {}
    json_list = {}
    frame_list = {}
    last_access = {}
    json_data = {}
    detector = None
    embedder = None
    recognizer = None
    le = None
    # is_ended = False

    def initialize(self, monitor, stream_url):
        if monitor not in Camera.thread_list:
            # start background frame thread
            thread = threading.Thread(target=self._thread, args=(
                stream_url,), kwargs={"monitor": monitor})
            thread.start()
            Camera.thread_list[str(monitor)] = thread

            # wait until frames start to be available
            # while monitor not in self.frame_list or self.frame_list[str(monitor)] is None:
            #     time.sleep(0)

    def __init__(self):
        print('[INFO] loading face detector...')
        if Camera.detector is None:
            protoPath = 'face_detection_model/deploy.prototxt'
            modelPath = 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'
            Camera.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        if Camera.embedder is None:
            # load our serialized face embedding model from disk
            print('[INFO] loading face recognizer...')
            Camera.embedder = cv2.dnn.readNetFromTorch(
                'openface_nn4.small2.v1.t7')

        if Camera.recognizer is None:
            # load the actual face recognition model along with the label encoder
            Camera.recognizer = pickle.loads(
                open('output/recognizer.pickle', 'rb').read())

        if Camera.le is None:
            Camera.le = pickle.loads(open('output/le.pickle', 'rb').read())

    # def get_frame(self, monitor):
    #     try:
    #         return self.frame_list[str(monitor)]
    #     except:
    #         return None

    def get_json(self, monitor):
        try:
            return self.json_list[str(monitor)]
        except:
            return {}

    def change_stream_url(self, monitor, stream_url):
        if monitor in Camera.thread_list:
            return None

        self.initialize(monitor, stream_url)

    @classmethod
    def _thread(cls, stream_url, monitor=0):
        # login to zm server first
        r = requests.post(url=LOGIN_URL)
        print('[INFO] openning video stream...')
        auth_info = r.json()['credentials']
        new_url = f'{ZM_STREAM_URL}?mode=jpeg&monitor={monitor}&{auth_info}'
        # start streaming ưith zm stream url
        cap = cv2.VideoCapture(new_url)
        if cap is None or not cap.isOpened():
            # try to open alternative url
            print('[ERROR] trying to open direct url...')
            cap = cv2.VideoCapture(stream_url)
            if cap is None or not cap.isOpened():
                print('[ERROR] unable to open remote stream...')
                cap.release
                cls.thread_list[str(monitor)] = None
                return
        # cap = cv2.VideoCapture(0)
        print('[INFO] starting face detection...')
        while True:
            try:
                response_data = {}
                response_data['detection'] = []
                ret, frame = cap.read()
                #ret, frame = camera.read()
                if not ret:
                    continue

                # resize the frame to have a width of 600 pixels (while
                # maintaining the aspect ratio), and then grab the image
                # dimensions
                # frame = imutils.resize(frame, width=600)
                (h, w) = frame.shape[:2]

                # construct a blob from the image
                # imageBlob = cv2.dnn.blobFromImage(
                #     cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                #     (104.0, 177.0, 123.0), swapRB=False, crop=False)
                imageBlob = cv2.dnn.blobFromImage(
                    frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

                # apply OpenCV's deep learning-based face detector to localize
                # faces in the input image
                cls.detector.setInput(imageBlob)
                detections = cls.detector.forward()

                # color = [255, 255, 255]
                # frame = cv2.copyMakeBorder(
                #     frame, 0, 0, 0, 200, cv2.BORDER_CONSTANT, value=color)

                # loop over the detections
                for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections
                    if confidence > 0.5:
                        # compute the (x, y)-coordinates of the bounding box for
                        # the face
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype('int')

                        # extract the face ROI
                        face = frame[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        # ensure the face width and height are sufficiently large
                        if fW < 20 or fH < 20:
                            continue

                        # construct a blob for the face ROI, then pass the blob
                        # through our face embedding model to obtain the 128-d
                        # quantification of the face
                        faceBlob = cv2.dnn.blobFromImage(
                            face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        cls.embedder.setInput(faceBlob)
                        vec = cls.embedder.forward()

                        # perform classification to recognize the face
                        preds = cls.recognizer.predict_proba(vec)[0]
                        j = np.argmax(preds)
                        proba = preds[j]
                        name = cls.le.classes_[j]

                        # draw the bounding box of the face along with the
                        # associated probability
                        # text = '{}: {:.2f}%'.format(name, proba * 100)
                        # text = f'{name}さんを検知しました。'
                        # y = startY - 10 if startY - 10 > 10 else startY + 10
                        # cv2.rectangle(frame, (startX, startY), (endX, endY),
                        #               (0, 0, 255), 2)
                        # cv2.putText(frame, text, (startX, y),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        # frame_pil = Image.fromarray(frame)
                        # draw = ImageDraw.Draw(frame_pil)
                        # font = ImageFont.truetype('hgrpp1.ttc', 12)
                        # draw.text((startX, y),  text,
                        #           font=font, fill=(0, 0, 0))
                        # frame = np.array(frame_pil)
                        # top = top + 20

                        json_data = {}
                        json_data['name'] = '{}'.format(name)
                        json_data['time'] = datetime.datetime.now().strftime(
                            '%Y-%m-%d %H:%M:%S')
                        json_data['confidence'] = str(confidence)
                        response_data['detection'].append(json_data)

                cls.json_list[str(monitor)] = response_data

                # ret, jpeg = cv2.imencode('.jpg', frame)
                # cls.frame_list[str(monitor)] = jpeg.tobytes()
            finally:
                time.sleep(0.25)

        print('[INFO] releasing stream resources...')
        cap.release
        cls.thread_list[str(monitor)] = None

    def detect_image(self, frame):
        response_data = {}
        response_data['detection'] = []
        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        try:
            Camera.detector.setInput(imageBlob)
            detections = Camera.detector.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI
                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue
                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    faceBlob = cv2.dnn.blobFromImage(
                        face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    Camera.embedder.setInput(faceBlob)
                    vec = Camera.embedder.forward()

                    # perform classification to recognize the face
                    preds = Camera.recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = Camera.le.classes_[j]

                    data = {}
                    data['name'] = name
                    data['time'] = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S")
                    data['confidence'] = str(proba)
                    response_data['detection'].append(data)
        finally:
            return response_data
