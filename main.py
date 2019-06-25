# Project: Face Detection System

import requests
from train_model import train_model
from extract_embeddings import extract_data
from detector_thread_faced import Camera
from flask import Flask, render_template, Response, request
import cv2
import numpy
import json

app = Flask(__name__)


def gen_response(response_data):
    resp = Response(json.dumps(response_data))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/view_json', methods=['POST'])
def view_json():
    response_data = {}
    response_data['detection'] = []
    # validate request data
    if 'id' not in request.form:
        return gen_response(response_data)

    monitor = request.form['id']
    detected_data = Camera().get_json(monitor)
    # get list of targeted persons
    target_person = []
    if 'targets' in request.form:
        target_person = request.form['targets']
    # if there is no target, so track all persons
    if len(target_person) <= 0:
        return gen_response(detected_data)

    # if there are some detected data
    if 'detection' in detected_data and len(detected_data['detection']) > 0:
        for data in detected_data['detection']:
            # if detected data in in targeted persons
            if data['name'] in target_person:
                response_data['detection'].append(data)

    return gen_response(response_data)


@app.route('/do_training', methods=['GET'])
def do_training():
    confidence = 0.5
    if 'confidence' in request.args:
        confidence = request.args.get('confidence')
    extract_data(confidence)
    train_model()
    return Response(json.dumps({'result': 'ok'}))


@app.route('/change_stream_url', methods=['POST'])
def change_stream_url():
    if 'monitor_id' not in request.get_json():
        return Response(json.dumps({'result': 'ng'}))

    stream_url = ''
    monitor = request.get_json()['monitor_id']
    if ('stream_url' in request.get_json()):
        stream_url = request.get_json()['stream_url']

    Camera().change_stream_url(monitor, stream_url)
    return Response(json.dumps({'result': 'ok'}))


@app.route('/valid_image', methods=['POST', 'PUT'])
def confirm_image():
    response_data = {}
    response_data['detection'] = []
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return gen_response(response_data)
        file = request.files['file']
        # check if file name is empty
        if file.filename == '':
            return gen_response(response_data)

        # check file validate
        # if file and allowed_file(file.filename):
        if file:
            filestr = request.files['file'].read()
            # convert string data to numpy array
            npimg = numpy.fromstring(filestr, numpy.uint8)
            # convert numpy array to image
            img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
            response = Camera().detect_image(img)
            return gen_response(response)

    return gen_response(response_data)


@app.route('/valid_event', methods=['POST', 'PUT'])
def confirm_video():
    response_data = {}
    response_data['detection'] = []
    if request.method != 'POST' or 'event_id' not in request.form or 'monitor_id' not in request.form or 'event_date' not in request.form:
        return gen_response(response_data)

    event_id = request.form['event_id']
    monitor_id = request.form['event_id']
    event_date = request.form['event_date']
    detected_data = Camera().detect_video(event_id, monitor_id, event_date)

    target_person = []
    if 'targets' in request.form:
        target_person = request.form['targets']
    # if there is no target, so track all persons
    if len(target_person) <= 0:
        return gen_response(detected_data)

    # if there are some detected data
    if 'detection' in detected_data and len(detected_data['detection']) > 0:
        for data in detected_data['detection']:
            # if detected data in in targeted persons
            if data in target_person:
                response_data['detection'].append(data)

    return gen_response(response_data)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port='5000', debug=True)
    app.run(host='0.0.0.0', threaded=True, ssl_context=('/home/ubuntu/ssl/server.crt', '/home/ubuntu/ssl/server.key'))
    #app.run(host='0.0.0.0', threaded=True)
