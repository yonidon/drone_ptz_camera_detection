from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import sqlite3
import requests
from requests.auth import HTTPDigestAuth
import logging
import time
import re
import cv2
import math
import configparser
import paramiko
from sqlalchemy import create_engine ,Table, MetaData, update , select, insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import func
from sqlalchemy.sql import delete, select
from sshtunnel import SSHTunnelForwarder
import pymysql
import json
import os

logging.basicConfig(level=logging.WARNING)

#Read config values
config = configparser.ConfigParser()
config.read('drone_config.ini')
conf_thresh = float(config['DEFAULT']['conf_thresh'])
stream_url = config['DEFAULT']['stream_url']
username = config['DEFAULT']['username']
password = config['DEFAULT']['password']
screen_width = int(config['DEFAULT']['screen_width'])
screen_height = int(config['DEFAULT']['screen_height'])
channel = int(config['DEFAULT']['channel'])
rtsp_url = f'rtsp://{username}:{password}@{stream_url}:554/cam/realmonitor?channel={channel}&subtype=1'
position = config['DEFAULT']['position']
fps = int(config['DEFAULT']['fps'])
tourNum = int(config['DEFAULT']['tourNum'])
server_port = int(config['DEFAULT']['server_port'])
# Load YOLOv5 model and Define the classes you want to detect
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/guard3/drone/best.pt', source='github')
classes = ['Drone']

# Set video source (webcam or video file), if not working, change subtype or check if camera is connected. For heat camera, change channel to 2
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_FPS, fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Flask app setup
app = Flask(__name__)

# Drone Bounding box center. Needed for the camera to move to the center.
def get_bounding_box_center(xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) // 2
    y_center = (ymin + ymax) // 2
    return x_center, y_center

# Function to clear buffer, reduces frame delay
def clear_buffer(cap, buffer_size=5):
    for i in range(buffer_size):
        cap.grab()

#Interpolation of FOV values - Calculates current hfov and current vfov
def interpolate(min_value, max_value, min_zoom, max_zoom, current_zoom):
    return min_value + (max_value - min_value) * ((current_zoom - min_zoom) / (max_zoom - min_zoom))

#Calculate the level of zoom adjustment needed towards drone
def calculate_zoom_level(x1, y1, x2, y2, screen_width, screen_height, current_zoom_level):
    bbox_area = (x2 - x1) * (y2 - y1)
    screen_area = screen_width * screen_height
    target_area = screen_area / 12
    zoom_adjustment_factor = math.sqrt( target_area / bbox_area )
    new_zoom_level = current_zoom_level * zoom_adjustment_factor
    print(f"Screen area{screen_area} Bbox Area{bbox_area} Target Area{target_area}")
    return new_zoom_level

# Calculate the DOA (Direction of Arrival) (add hfov and vfov)
def calculate_doa(current_hfov,current_vfov,current_pan_position, current_tilt_position, center_x, center_y, screen_width, screen_height):
    # Calculate the offset of the bounding box center from the frame center
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2
    offset_x = center_x - screen_center_x
    offset_y = center_y - screen_center_y
    
    # Calculate the angle offsets
    angle_offset_x = (offset_x / screen_width) * current_hfov
    angle_offset_y = (offset_y / screen_height) * current_vfov
    
    # Calculate the absolute DOA based on the current pan and tilt angles
    doa_pan = current_pan_position + angle_offset_x
    doa_tilt = current_tilt_position + angle_offset_y
    
    # Normalize the DOA angles
    doa_pan = (doa_pan + 360) % 360  # Ensure the angle is within 0-360 degrees
    doa_tilt = max(min(doa_tilt, 90), -90)  # Ensure the tilt angle is within -90 to 90 degrees
    
    return doa_pan, doa_tilt


#Initiate Global variable for DOA
latest_doa_pan = 0
latest_doa_tilt = 0

# Generate frames for the video feed, analyze image for drones and move camera towards drones
def gen_frames():
    #Use the global doa variable
    global latest_doa_pan 
    global latest_doa_tilt 
    #Instantiate the time of last zoom commands sent to camera, this is to prevent the camera from zooming too much
    last_zoom_time = time.time()
    #Loop of frame captures
    while True:
        # Clear the buffer to get the latest frame, this reduces delay
        clear_buffer(cap)
        
        # Read frame from video source
        ret, frame = cap.read()

        # Will stop process if no feed
        if not ret:
            break

        # Resize window so it won't be too big
        frame_height, frame_width = frame.shape[:2]
        frame_resized = cv2.resize(frame, (screen_width, screen_height))

        # Convert the frame to a format that YOLOv5 can process
        img = Image.fromarray(frame[..., ::-1])

        # Run inference on the frame
        results = model(img, size=640)

        # Drone is detected, this loop is processing the results and drawing bounding boxes on the frame
        flag_exist_drone=0
        for result in results.xyxy[0]:
            #Coordinates of drone location on screen
            x1, y1, x2, y2, conf, cls = result.tolist()
            global conf_thresh
            #Drone is detected if above confidence score(conf)
            if conf > conf_thresh and classes[int(cls)] in classes:
                # Scale bounding box coordinates to resized frame
                x1 = int(x1 * screen_width / frame_width)
                y1 = int(y1 * screen_height / frame_height)
                x2 = int(x2 * screen_width / frame_width)
                y2 = int(y2 * screen_height / frame_height)

                # Drone detection time, append to data frame
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Clear the buffer to get the latest frame, this reduces delay
                clear_buffer(cap)

                # Draw the bounding box
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Display the confidence score above the box
                text_conf = "{:.2f}%".format(conf * 100)

                # In this section, the camera moves to the direction of the drone
                center_x, center_y = get_bounding_box_center(x1, y1, x2, y2)

                ###################Get Values from drone status page#########
                global stream_url
                global username
                global password
                url = f'http://{stream_url}/cgi-bin/ptz.cgi?action=getStatus'
                response = requests.get(url, auth=HTTPDigestAuth(username, password))
                # Get Camera zoom, pan angle hd, and tilt angle hd
                camera_zoom = int(re.search(r'status\.ZoomValue=(\d+)', str(response.content)).group(1))
                current_pan_position = int(re.search(r'status\.PanAngleHD=(-?\d+)', str(response.content)).group(1))
                current_tilt_position = int(re.search(r'status\.TiltAngleHD=(-?\d+)', str(response.content)).group(1))
                ################Interpolation of FOV###################
                horizontal_fov = 61.8  # degrees, at current zoom level
                vertical_fov = 37.2    # degrees, at current zoom level
                min_zoom = 10
                max_zoom = 370
                min_hfov = 61.8  # degrees
                max_hfov = 1.86  # degrees
                min_vfov = 37.2  # degrees
                max_vfov = 1.05  # degrees
                min_focal_length = 6.5  # mm
                max_focal_length = 240  # mm                
                current_hfov = interpolate(min_hfov, max_hfov, min_zoom, max_zoom, camera_zoom)
                current_vfov = interpolate(min_vfov, max_vfov, min_zoom, max_zoom, camera_zoom)
                ################Calculate Doa##########################
                doa_pan,doa_tilt = calculate_doa(current_hfov,current_vfov,current_pan_position, current_tilt_position, center_x, center_y, screen_width, screen_height)
                latest_doa_pan = doa_pan
                latest_doa_tilt = doa_tilt
                print(f"DOA:{latest_doa_pan} and {latest_doa_tilt}")

                ###############Guardian Data Insertion#################
                '''
                db_host = '127.0.0.1'  # Localhost on the remote server
                db_user = 'sgb'
                db_password = 'sgb'
                db_name = 'sgb'
                db_port = 3306
                connection_string = f'mysql+pymysql://{db_user}:{db_password}@127.0.0.1:3309/{db_name}'

                # Create a SQLAlchemy engine
                engine = create_engine(connection_string)
                metadata = MetaData()
                activity_table = Table('TBL_ACTIVITY', metadata, autoload_with=engine)

                # Fetch the highest EVENT_ID
                with engine.connect() as connection:
                    max_event_id_query = select(func.max(activity_table.c.EVENT_ID))
                    result = connection.execute(max_event_id_query).fetchone()
                    max_event_id = result[0] if result[0] is not None else 0
                    new_event_id = max_event_id + 1
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Data to insert
                data = {
                    "IMSI": ["42502c8583a9c"],
                    "IMEI": [None],
                    "IMSI_NAME": ["Camera UAV"],
                    "IMEI_NAME": [None],
                    "LIST": [None],
                    "STATUS": ["Passive"],
                    "LAST_UPDATE": [current_time],
                    "EVENT_ID": [new_event_id],
                    "MSISDN": [None],
                    "B_PARTY": [None],
                    "RX_LEVEL": [75],
                    "POPUP": [0],
                    "BTS_ID": [16],
                    "ORIG_BTS_ID": [0],
                    "TAPD": [0],
                    "GROUP_ID": [4],
                    "GPRS_STATUS": [None],
                    "IP": [None],
                    "LATITUDE": [0],
                    "LONGITUDE": [99999],
                    "ACCURACY": [None],
                    "TMSI": [None],
                    "IMEI_SV": [None],
                    "CIPHER_MODE": [0],
                    "DL_RX_LEVEL": [0],
                    "UL_RX_QUAL": [36.7],
                    "DL_RX_QUAL": [0],
                    "DISTANCE": [0],
                    "ACTIVE": [1],
                    "NET_STATUS": [None],
                    "ORIG_LAC": [0],
                    "ATT_STATUS": [None],
                    "LOCATION": [json.dumps({"type_1": {"time": f"{current_time}", "areas": [{"rad_a": [1382, 1582], "rad_b": [1382, 1582], "angles": [latest_doa_pan, latest_doa_tilt, 0], "coords": [32.09811, 34.84780, 0.0]}]}})],
                    "WIFI": [json.dumps({})],
                    "BT": [json.dumps({})],
                    "FIRST_SEEN": [current_time],
                    "RAT": [10]
                }

                # Create a DataFrame
                df = pd.DataFrame(data)

                # Insert data into the table
                try:
                    #df.to_sql('TBL_ACTIVITY', con=engine, if_exists='append', index=False)
                    print("Data inserted successfully with EVENT_ID:", new_event_id)
                except Exception as e:
                    print("Error: ",e)  
                    '''
                ###############Calculate Requried Camera Position#############
                screen_center_x = screen_width // 2
                screen_center_y = screen_height // 2
                offset_x = center_x - screen_center_x
                offset_y = center_y - screen_center_y 
                pan_degrees = (offset_x / screen_width) * current_hfov
                tilt_degrees = (offset_y / screen_height) * current_vfov
                pan_position = 0 + pan_degrees * 100 #Should have used current_pan_position instead of 0 but not working well
                tilt_position = 0 + tilt_degrees * 100

                #############Calculation of Zoom Level#################
                #This will calculate how much zoom is needed towards the drone
                zoom_adjustment = 0
                #Note: if you put an integer in args3 it will increment that zoom by that integer
                current_time = time.time()
                print(f"{last_zoom_time}")
                if current_time - last_zoom_time >= 6:
                   desired_zoom_level = calculate_zoom_level(x1, y1, x2, y2, screen_width, screen_height, camera_zoom)
                   zoom_adjustment = int(desired_zoom_level-camera_zoom) 
                   last_zoom_time = current_time
                   print(f"Camera Zoom: {camera_zoom} and Zoom Adjustment: {zoom_adjustment} and Desired zoom level: {desired_zoom_level}")

                ##################Move Camera Towards Drone##################
                #args3 is 0 because zoom is not working so well
                payload_position = {'action': 'start', 'channel': channel, 'code': 'Position', 'arg1': int(pan_position), 'arg2': int(tilt_position), 'arg3': 0}
                response_position = requests.get(f'http://{stream_url}/cgi-bin/ptz.cgi?', params=payload_position, auth=HTTPDigestAuth(username, password))
                print("Moving Camera: " + str(response_position.content))
                
                ##################Draw Bounding box over drone###############
                cv2.putText(frame_resized, text_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                text_coords = "({}, {})".format(int((x1 + x2) / 2), int(y2))
                cv2.putText(frame_resized, text_coords, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame_resized)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to get the latest DOA value
@app.route('/get_doa')
def get_doa():
    global latest_doa_pan, latest_doa_tilt
    logging.debug(f"Returning DOA: {latest_doa_pan} and {latest_doa_tilt}")
    return jsonify({'doa_pan': latest_doa_pan, 'doa_tilt': latest_doa_tilt})
#Endpoint to move camera
@app.route('/move_camera', methods=['POST'])
def move_camera():
    data = request.json
    direction = data.get('direction')
    action = data.get('action')
    code = ''

    if direction == 'up':
        code = 'Up'
    elif direction == 'down':
        code = 'Down'
    elif direction == 'left':
        code = 'Left'
    elif direction == 'right':
        code = 'Right'

    if code:
        payload_position = {'action': action, 'channel': channel, 'code': code, 'arg1': '0', 'arg2': '1', 'arg3': '0'}
        response_position = requests.get(f'http://{stream_url}/cgi-bin/ptz.cgi?', params=payload_position, auth=HTTPDigestAuth(username, password))
        return jsonify({'status': 'success'}), 200
    else:
        return jsonify({'status': 'invalid direction'}), 400
#Endpoint to zoom camera
@app.route('/zoom_camera', methods=['POST'])
def zoom_camera():
    data = request.json
    zoomType = data.get('zoomType')
    
    if zoomType in ['ZoomTele', 'ZoomWide']:
        payload_zoom = {'action': 'start', 'channel': channel, 'code': zoomType, 'arg1': '0', 'arg2': '1', 'arg3': '0'}
        response_zoom = requests.get(f'http://{stream_url}/cgi-bin/ptz.cgi?', params=payload_zoom, auth=HTTPDigestAuth(username, password))
        return jsonify({'status': 'success'}), 200
    else:
        return jsonify({'status': 'invalid zoom type'}), 400
#Endpoint to automatic scan mode    
@app.route('/tour', methods=['POST'])
def tour():
    data = request.json
    action = data.get('action')
    
    if action not in ['start', 'stop']:
        return jsonify({'status': 'invalid action'}), 400

    tourCode = 'StartScanTour' if action == 'start' else 'StartTour' #StartScanTour is code for starting scan, StartTour is for stopping. It's because of strange api
    payload_tour = {
        'action': 'start',
        'channel': channel,
        'code': tourCode,
        'arg1': tourNum,
        'arg2': 0,
        'arg3': 3,
        'arg4': 0
    }
    try:
        response_tour = requests.get(
            f'http://{stream_url}/cgi-bin/ptz.cgi?',
            params=payload_tour,
            auth=HTTPDigestAuth(username, password)
        )
        response_tour.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.exceptions.RequestException as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

    return jsonify({'status': 'success'}), 200
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=server_port)
