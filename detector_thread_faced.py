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

from faced import FaceDetector
from faced.utils import annotate_image

from config_reader import read_config

ZM_URL = 'http://18.179.207.49/zm'
ZM_STREAM_URL = f'{ZM_URL}/cgi-bin/nph-zms'
LOGIN_URL = f'{ZM_URL}/api/host/login.json?user=admin&pass=admin'


def connect_stream(monitor, stream_url):
    r = requests.post(url=LOGIN_URL)
    print('[INFO] openning video stream...')
    auth_info = r.json()['credentials']
    new_url = f'{ZM_STREAM_URL}?mode=jpeg&maxfps=5&monitor={monitor}&{auth_info}'
    # start streaming with zm stream url
    cap = cv2.VideoCapture(new_url)
    if cap is None or not cap.isOpened():
        # try to open alternative url
        print('[ERROR] trying to open direct url...')
        cap = cv2.VideoCapture(stream_url)
    return cap


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
    max_retry_count = 0
    stream_url_list = {}

    confidence = 0.90
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
        file_paths, configs = read_config()
        if Camera.detector is None:
            print('[INFO] loading face detector...')
            Camera.detector = FaceDetector()

        if Camera.embedder is None:
            # load our serialized face embedding model from disk
            print('[INFO] loading embedder from {}'.format(
                file_paths['embedder_path']))
            Camera.embedder = cv2.dnn.readNetFromTorch(
                file_paths['embedder_path'])

        if Camera.recognizer is None:
            # load the actual face recognition model along with the label encoder
            print('[INFO] loading face recognizer from {}'.format(
                file_paths['recognizer_path']))
            Camera.recognizer = pickle.loads(
                open('output/recognizer.pickle', 'rb').read())

        if Camera.le is None:
            print('[INFO] loading le from {}'.format(file_paths['le_path']))
            Camera.le = pickle.loads(open('output/le.pickle', 'rb').read())

        print('[INFO] Confidence value is set to {}'.format(
            configs['confidence']))
        Camera.confidence = float(configs['confidence'])

        Camera.max_retry_count = int(configs['max_retry_count'])

    # def get_frame(self, monitor):
    #     try:
    #         return self.frame_list[str(monitor)]
    #     except:
    #         return None

    def get_json(self, monitor):
        try:
            return self.json_list[str(monitor)]
        except:
            response_data = {}
            response_data['detection'] = []
            return response_data

    def change_stream_url(self, monitor, stream_url):
        if monitor in Camera.thread_list:
            return None

        Camera.stream_url_list[str(monitor)] = stream_url
        self.initialize(monitor, stream_url)

    @classmethod
    def _thread(cls, stream_url, monitor=0):
        # login to zm server first
        r = requests.post(url=LOGIN_URL)
        print('[INFO] openning video stream...')
        auth_info = r.json()['credentials']
        new_url = f'{ZM_STREAM_URL}?mode=jpeg&maxfps=5&monitor={monitor}&{auth_info}'

        retry_count = 0
        cap = None
        # start trying to connect to streaming resource
        while (cap is None or not cap.isOpened) and retry_count < cls.max_retry_count:
            cap = connect_stream(monitor, cls.stream_url_list[str(monitor)])
            retry_count += 1
        if cap is None or not cap.isOpened():
            print('[ERROR] unable to open remote stream...')
            cls.thread_list[str(monitor)] = None
            return

        print('[INFO] starting face detection...')
        cap_failed_count = 0
        while True:
            try:
                response_data = {}
                response_data['detection'] = []
                ret, frame = cap.read()
                #ret, frame = camera.read()
                if not ret:
                    cap_failed_count += 1
                    cls.json_list[str(monitor)] = response_data

                    if (cap_failed_count > cls.max_retry_count):
                        if cap.isOpened():
                            cap.release()
                        retry_count = 0
                        while (cap is None or not cap.isOpened) and retry_count < cls.max_retry_count:
                            cap = connect_stream(
                                monitor, cls.stream_url_list[str(monitor)])
                            retry_count += 1
                        if cap is None or not cap.isOpened():
                            print('[ERROR] unable to open remote stream...')
                            cls.thread_list[str(monitor)] = None
                            return
                    continue

                # resize the frame to have a width of 600 pixels (while
                # maintaining the aspect ratio), and then grab the image
                # dimensions
                frame = imutils.resize(frame, width=600)
                # (h, w) = frame.shape[:2]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bboxes = cls.detector.predict(frame, cls.confidence)

                # ensure at least one face was found
                print('[INFO] detected faces: {}'.format(len(bboxes)))
                if len(bboxes) > 0:
                    for xb, yb, wb, hb, pb in bboxes:
                        startX = int(xb - wb/2)
                        startY = int(yb - hb/2)
                        endX = int(xb + wb/2)
                        endY = int(yb + hb/2)

                        # extract the face ROI
                        face = frame[startY:endY, startX:endX]
                        # (fH, fW) = face.shape[:2]

                        # ensure the face width and height are sufficiently large
                        # if fW < 20 or fH < 20:
                        #     continue

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
                        # name = 0
                        # if proba >= 0.6:
                        #     name = cls.le.classes_[j]

                        json_data = {}
                        json_data['name'] = '{}'.format(name)
                        json_data['time'] = datetime.datetime.now().strftime(
                            '%Y-%m-%d %H:%M:%S')
                        json_data['confidence'] = str(proba)
                        response_data['detection'].append(json_data)

                cls.json_list[str(monitor)] = response_data

                # ret, jpeg = cv2.imencode('.jpg', frame)
                # cls.frame_list[str(monitor)] = jpeg.tobytes()
            finally:
                time.sleep(0.02)

        print('[INFO] releasing stream resources...')
        if cap.isOpened():
            cap.release()
        cls.thread_list[str(monitor)] = None

    def detect_image(self, frame):
        response_data = {}
        response_data['detection'] = []
        response_list = []
        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        # frame = imutils.resize(frame, width=600)
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes = Camera.detector.predict(frame, Camera.confidence)

            # ensure at least one face was found
            print('[INFO] detected faces: {}'.format(len(bboxes)))
            if len(bboxes) > 0:
                for xb, yb, wb, hb, pb in bboxes:
                    startX = int(xb - wb/2)
                    startY = int(yb - hb/2)
                    endX = int(xb + wb/2)
                    endY = int(yb + hb/2)

                    # extract the face ROI
                    face = frame[startY:endY, startX:endX]
                    # (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    # if fW < 20 or fH < 20:
                    #     continue

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
                    # name = 0
                    # if proba >= 0.6:
                    #     name = Camera.le.classes_[j]

                    if name not in response_list:
                        response_list.append(name)
                    json_data = {}
                    json_data['name'] = '{}'.format(name)
                    json_data['time'] = datetime.datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S')
                    json_data['confidence'] = str(proba)
                    response_data['detection'].append(json_data)
        finally:
            return response_data, response_list

    def detect_video(self, event_id, monitor_id, event_date):
        response_data = {}
        response_data['detection'] = []
        # cap = cv2.VideoCapture(0)
        print(f'[INFO] starting face detection for event {event_id}...')
        result_list = []
        start_index = 1
        while(True):
            img_path = f'/mnt/zoneminder/events/{monitor_id}/{event_date}/{event_id}/{start_index:05}-capture.jpg'
            try:
                print(f'[INFO] parsing {img_path}...')
                ret, frame = cv2.imread(img_path)
                if not ret:
                    break
                detect_data, detect_list = self.detect_image(frame)
                for detect_id in detect_list:
                    if detect_id not in result_list:
                        result_list.append(detect_id)
            except Exception as e:
                print(e)
                print(
                    f'[INFO] failed to parsing frame {start_index} for event {event_id}...')
                break
            finally:
                start_index += 1
                time.sleep(0.02)
        print('[INFO] finish video detection...')
        response_data['detection'] = result_list
        return response_data
