#!/usr/bin/python

from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from multiprocessing import Process, Queue
import sys
import time
import math
import cv2
import atexit
import pyrealsense2 as rs
import numpy as np
import traceback

import sqlite3
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, logUsage, cudaFromNumpy, cudaAllocMapped, cudaConvertColor


network = "ssd-mobilenet-v2"
threshold = 0.5
width = 640
height = 480

m1_top = m2_top = m3_top = m4_top = m5_top = height // 3
m1_bottom = m2_bottom = m3_bottom = m4_bottom = m5_bottom = 2*height // 3
    
m1_left = 0
m1_right = m2_left = width // 5
m2_right = m3_left = 2*width // 5
m3_right = m4_left = 3*width // 5
m5_left = m4_right = 4*width // 5
m5_right = width



#Need app-scoped place to put the distance data

def connect_to_db():
    conn = sqlite3.connect('database.db')
    return conn


def create_db_tables():
    try:
        conn = connect_to_db()
        conn.execute('''DROP TABLE IF EXISTS distances''')
        conn.execute('''
            CREATE TABLE distances (
                id INTEGER PRIMARY KEY NOT NULL,
                frame_id INTEGER NOT NULL,
                object_name TEXT NOT NULL,
                distance REAL NOT NULL,
                left INTEGER NOT NULL,
                top INTEGER NOT NULL,
                right INTEGER NOT NULL,
                bottom INTEGER NOT NULL
            );
        ''')

        conn.commit()
        print("[distances] table created successfully")
    except:
        print(traceback.format_exc())
        print("[distances] table creation failed")
    finally:
        conn.close()

    try:
        conn = connect_to_db()
        conn.execute('''DROP TABLE IF EXISTS averages''')
        conn.execute('''
            CREATE TABLE averages (
                id INTEGER PRIMARY KEY NOT NULL,
                frame_id INTEGER NOT NULL,
                m1_avg_depth REAL NOT NULL,
                m1_avg_close_depth REAL NOT NULL,
                m2_avg_depth REAL NOT NULL,
                m2_avg_close_depth REAL NOT NULL,
                m3_avg_depth REAL NOT NULL,
                m3_avg_close_depth REAL NOT NULL,
                m4_avg_depth REAL NOT NULL,
                m4_avg_close_depth REAL NOT NULL,
                m5_avg_depth REAL NOT NULL,
                m5_avg_close_depth REAL NOT NULL
            );
        ''')

        conn.commit()
        print("[averages] table created successfully")
    except:
        print(traceback.format_exc())
        print("[averages] table creation failed")
    finally:
        conn.close()

def insert_distance(ds):
    distance_info = {}
    try:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO distances (frame_id, object_name, distance, left, top, right, bottom)" +
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", 
                    (ds['frame_id'], 
                     ds['object_name'], 
                     ds['distance'], 
                     ds['left'], 
                     ds['top'], 
                     ds['right'], 
                     ds['bottom']) )
        conn.commit()
        distance_info = get_distance_by_id(cur.lastrowid)
    except:
        conn().rollback()

    finally:
        conn.close()

    distances = get_distances()
    if len(distances) != 0:

        latest_id = distances[len(distances) - 1]['id']
        #print('latest_id', latest_id)

        if len(distances) > 30:
            delete_old_distances(latest_id)


    return distance_info

def insert_averages(avs):
    #print(avs)
    averages_info = {}
    try:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO averages (frame_id, m1_avg_depth, m1_avg_close_depth,   \
                    m2_avg_depth, m2_avg_close_depth, m3_avg_depth, m3_avg_close_depth,  \
                    m4_avg_depth, m4_avg_close_depth, m5_avg_depth, m5_avg_close_depth)  \
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",                       
                    (avs['frame_id'],                                                
                     avs['m1_avg_depth'], avs['m1_avg_close_depth'],                 
                     avs['m2_avg_depth'], avs['m2_avg_close_depth'],                 
                     avs['m3_avg_depth'], avs['m3_avg_close_depth'],                 
                     avs['m4_avg_depth'], avs['m4_avg_close_depth'],                 
                     avs['m5_avg_depth'], avs['m5_avg_close_depth']) )
        conn.commit()
        averages_info = get_averages_by_id(cur.lastrowid)

    except:
        conn().rollback()

    finally:
        conn.close()

    #do some garbage collection so we don't kill the memory
    averages = get_averages()
    if len(averages) != 0:
    
        latest_id = averages[len(averages) - 1]['id']
        #print('latest_id', latest_id)
    
        if len(averages) > 30:
            delete_old_averages(latest_id)


    return averages_info

def get_distances():
    dss = []
    try:
        conn = connect_to_db()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM distances")
        rows = cur.fetchall()

        # convert row objects to dictionary
        for i in rows:
            ds = {}
            ds["id"] = i["id"]
            ds["frame_id"] = i["frame_id"]
            ds["object_name"] = i["object_name"]
            ds["distance"] = i["distance"]
            ds["left"] = i["left"]
            ds["top"] = i["top"]
            ds["right"] = i["right"]
            ds["bottom"] = i["bottom"]
            dss.append(ds)

    except:
        dss = []

    return dss

def get_averages():
    avs = []
    try:
        conn = connect_to_db()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM averages")
        rows = cur.fetchall()

        # convert row objects to dictionary
        for i in rows:
            av = {}
            av["id"] = i["id"]
            av["frame_id"] = i["frame_id"]
            av["m1_avg_depth"] = i["m1_avg_depth"]
            av["m1_avg_close_depth"] = i["m1_avg_close_depth"]
            av["m2_avg_depth"] = i["m2_avg_depth"]
            av["m2_avg_close_depth"] = i["m2_avg_close_depth"]
            av["m3_avg_depth"] = i["m3_avg_depth"]
            av["m3_avg_close_depth"] = i["m3_avg_close_depth"]
            av["m4_avg_depth"] = i["m4_avg_depth"]
            av["m4_avg_close_depth"] = i["m4_avg_close_depth"]
            av["m5_avg_depth"] = i["m5_avg_depth"]
            av["m5_avg_close_depth"] = i["m5_avg_close_depth"]
            avs.append(av)

    except:
        avs = []

    return avs



def get_most_recent_frame_info():
    av = {}
    frame_id = {}
    dss = []
    try:
        conn = connect_to_db()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM averages WHERE frame_id = (SELECT max(frame_id) from averages)")
        row = cur.fetchone()

        # convert row object to dictionary
        av["id"] = row["id"]
        av["frame_id"] = row["frame_id"]
        av["m1_avg_depth"] = row["m1_avg_depth"]
        av["m1_avg_close_depth"] = row["m1_avg_close_depth"]
        av["m2_avg_depth"] = row["m2_avg_depth"]
        av["m2_avg_close_depth"] = row["m2_avg_close_depth"]
        av["m3_avg_depth"] = row["m3_avg_depth"]
        av["m3_avg_close_depth"] = row["m3_avg_close_depth"]
        av["m4_avg_depth"] = row["m4_avg_depth"]
        av["m4_avg_close_depth"] = row["m4_avg_close_depth"]
        av["m5_avg_depth"] = row["m5_avg_depth"]
        av["m5_avg_close_depth"] = row["m5_avg_close_depth"]

        frame_id = row["frame_id"]

        cur.execute("SELECT * FROM distances WHERE frame_id = ?", (frame_id,))
        rows = cur.fetchall()

        for i in rows:
            ds = {}
            ds["id"] = i["id"]
            ds["frame_id"] = i["frame_id"]
            ds["object_name"] = i["object_name"]
            ds["distance"] = i["distance"]
            ds["left"] = i["left"]
            ds["top"] = i["top"]
            ds["right"] = i["right"]
            ds["bottom"] = i["bottom"]
            dss.append(ds)


    finally:
        conn.close()
    
    #print ('av',av)
    #print ('dss',dss)

    return av, dss


def get_distance_by_id(d_id):
    ds = {}
    try:
        conn = connect_to_db()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM distances WHERE id = ?", (did,))
        row = cur.fetchone()

        # convert row object to dictionary
        ds["id"] = row["id"]
        ds["frame_id"] = row["frame_id"]
        ds["object_name"] = row["object_name"]
        ds["distance"] = row["distnace"]
        ds["left"] = row["left"]
        ds["top"] = row["top"]
        ds["right"] = row["right"]
        ds["bottom"] = row["bottom"]

    except:
        ds = {}

    return ds

def get_averages_by_id(a_id):
    av = {}
    try:
        conn = connect_to_db()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM averages WHERE id = ?", (did,))
        row = cur.fetchone()

        # convert row object to dictionary
        av["id"] = row["id"]
        av["frame_id"] = row["frame_id"]
        av["m1_avg_depth"] = row["m1_avg_depth"]
        av["m1_avg_close_depth"] = row["m1_avg_close_depth"]
        av["m2_avg_depth"] = row["m2_avg_depth"]
        av["m2_avg_close_depth"] = row["m2_avg_close_depth"]
        av["m3_avg_depth"] = row["m3_avg_depth"]
        av["m3_avg_close_depth"] = row["m3_avg_close_depth"]
        av["m4_avg_depth"] = row["m4_avg_depth"]
        av["m4_avg_close_depth"] = row["m4_avg_close_depth"]
        av["m5_avg_depth"] = row["m5_avg_depth"]
        av["m5_avg_close_depth"] = row["m5_avg_close_depth"]

    except:
        av = {}

    return av

def delete_distance(d_id):
    message = {}
    try:
        conn = connect_to_db()
        conn.execute("DELETE from distances WHERE id = ?", (d_id,))
        conn.commit()
        message["status"] = "Distance deleted successfully"
    except:
        conn.rollback()
        message["status"] = "Cannot delete distance"
    finally:
        conn.close()

    return message


def delete_averages(a_id):
    message = {}
    try:
        conn = connect_to_db()
        conn.execute("DELETE from averages WHERE id = ?", (a_id,))
        conn.commit()
        message["status"] = "Averages deleted successfully"
    except:
        conn.rollback()
        message["status"] = "Cannot delete averages"
    finally:
        conn.close()

    return message


def delete_old_distances(d_id):
    message = {}
    KEEP = 20
    try:
        conn = connect_to_db()
        conn.execute("DELETE from distances WHERE id < ?", (d_id - KEEP,))
        conn.commit()
        message["status"] = "Distances deleted successfully"
    except:
        conn.rollback()
        message["status"] = "Cannot delete distance"
    finally:
        conn.close()

    return message

def delete_old_averages(a_id):
    message = {}
    KEEP = 20
    try:
        conn = connect_to_db()
        conn.execute("DELETE from averages WHERE id < ?", (a_id - KEEP,))
        conn.commit()
        message["status"] = "Averages deleted successfully"
    except:
        conn.rollback()
        message["status"] = "Cannot delete averages"
    finally:
        conn.close()

    return message

create_db_tables()



NUM_CLASSES = 90
classNames = ("human","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","street sign","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","hat","backpack","umbrella","shoe","eye glasses","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","plate","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","mirror","dining table","window","desk","toilet","door","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","blender","book","clock","vase","scissors","teddy bear","hair drier","toothbrush","hair brush")




def close_running_threads():
   print("Exit")
   pipeline.stop()

#Register the function to be called on exit
atexit.register(close_running_threads)


try:

    # load the object detection network
    net = detectNet(network, sys.argv, threshold)


    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    
    #device = profile.get_device()
    #device.hardware_reset()
        
    align_to = rs.stream.color
    align = rs.align(align_to)

except Exception:
    print(traceback.format_exc())





async_mode = None
async_mode = 'threading'
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app, async_mode=async_mode) 

@app.route('/')
def index():
    return render_template('index.html')

def inference_stream():

    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    
    depth_frame = aligned_frames.get_depth_frame() 
    color_frame = aligned_frames.get_color_frame()


    if not depth_frame or not color_frame:
        return


    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())


    # convert to CUDA (cv2 images are numpy arrays, in BGR format)
    bgr_img = cudaFromNumpy(color_image, isBGR=True)
    # convert from BGR -> RGB
    img = cudaAllocMapped(width=bgr_img.width,height=bgr_img.height,format='rgb8')
    cudaConvertColor(bgr_img, img)

    # detect objects in the image 
    detections = net.Detect(img)

    frame_number = rs.frame.get_frame_number(color_frame)



    # depth averages info
    top = depth_image[0:height//3, 0:width]
    middle = depth_image[height//3:2*height//3, 0:width]
    bottom = depth_image[2*height//3:height, 0:width]

    #m1_top = m2_top = m3_top = m4_top = m5_top = height // 3
    #m1_bottom = m2_bottom = m3_bottom = m4_bottom = m5_bottom = 2*height // 3
    #
    #m1_left = 0
    #m1_right = m2_left = width // 5
    #m2_right = m3_left = 2*width // 5
    #m3_right = m4_left = 3*width // 5
    #m5_left = m4_right = 4*width // 5
    #m5_right = width

    m1 = middle[0:-1, m1_left:m1_right]
    m2 = middle[0:-1, m2_left:m2_right]
    m3 = middle[0:-1, m3_left:m3_right]
    m4 = middle[0:-1, m4_left:m4_right]
    m5 = middle[0:-1, m5_left:m5_right]

    m1_avg_depth = np.average(m1).item()
    m2_avg_depth = np.average(m2).item()
    m3_avg_depth = np.average(m3).item()
    m4_avg_depth = np.average(m4).item()
    m5_avg_depth = np.average(m5).item()

    #avg of closest quarter of pixels
    closer_pixels = m1.shape[0] * m1.shape[1] // 4
   
    m1_avg_close_depth = np.average(np.partition(m1, kth=closer_pixels, axis=None)[:closer_pixels]).item()
    m2_avg_close_depth = np.average(np.partition(m2, kth=closer_pixels, axis=None)[:closer_pixels]).item()
    m3_avg_close_depth = np.average(np.partition(m3, kth=closer_pixels, axis=None)[:closer_pixels]).item()
    m4_avg_close_depth = np.average(np.partition(m4, kth=closer_pixels, axis=None)[:closer_pixels]).item()
    m5_avg_close_depth = np.average(np.partition(m5, kth=closer_pixels, axis=None)[:closer_pixels]).item()

    av = {}
    av["frame_id"] = frame_number
    av["m1_avg_depth"] = round(m1_avg_depth)
    av["m2_avg_depth"] = round(m2_avg_depth)
    av["m3_avg_depth"] = round(m3_avg_depth)
    av["m4_avg_depth"] = round(m4_avg_depth)
    av["m5_avg_depth"] = round(m5_avg_depth)
    av["m1_avg_close_depth"] = round(m1_avg_close_depth)
    av["m2_avg_close_depth"] = round(m2_avg_close_depth)
    av["m3_avg_close_depth"] = round(m3_avg_close_depth)
    av["m4_avg_close_depth"] = round(m4_avg_close_depth)
    av["m5_avg_close_depth"] = round(m5_avg_close_depth)
    
    insert_averages(av)

    m1_border = 5 if m1_avg_depth < 1000 else 1
    m2_border = 5 if m2_avg_depth < 1000 else 1
    m3_border = 5 if m3_avg_depth < 1000 else 1
    m4_border = 5 if m4_avg_depth < 1000 else 1
    m5_border = 5 if m5_avg_depth < 1000 else 1

    m1_border = 10 if m1_avg_close_depth < 500 else m1_border 
    m2_border = 10 if m2_avg_close_depth < 500 else m2_border
    m3_border = 10 if m3_avg_close_depth < 500 else m3_border
    m4_border = 10 if m4_avg_close_depth < 500 else m4_border
    m5_border = 10 if m5_avg_close_depth < 500 else m5_border
   
    # distances info

    for num in range(len(detections)) :
        score = round(detections[num].Confidence,2)
        box_top=int(detections[num].Top)
        box_left=int(detections[num].Left)
        box_bottom=int(detections[num].Bottom)
        box_right=int(detections[num].Right)
        box_center=detections[num].Center
        label = net.GetClassDesc(detections[num].ClassID)
        
        if (label == 'person'):
            label = 'human'
        if (label == 'bird'):
            label = 'CHICKEN'
        
        
        point_distance = np.round(depth_frame.get_distance(int(box_center[0]),int(box_center[1])), 3)
        distance_text = str(point_distance) + 'm'
     

        cv2.rectangle(color_image,(box_left,box_top),(box_right,box_bottom),(255,255,255),2)
        cv2.line(color_image,
            	(int(box_center[0])-10, int(box_center[1])),
            	(int(box_center[0]+10), int(box_center[1])),
            	(255, 255, 255), 3)
        cv2.line(color_image,
            	(int(box_center[0]), int(box_center[1]-10)),
            	(int(box_center[0]), int(box_center[1]+10)),
            	(255, 255, 255), 3)
        cv2.putText(color_image,
            	label + ' ' + distance_text,
            	(box_left+5,box_top+20),cv2.FONT_HERSHEY_SIMPLEX,0.4,
            	(255,255,255),1,cv2.LINE_AA)
            
        print("Detected a {0} {1:.3} meters away.".format(label, point_distance))

        ds = {}
        ds["object_name"] = label
        ds["frame_id"] = frame_number
        ds["distance"] = point_distance
        ds["left"] = box_left 
        ds["top"] = box_top
        ds["right"] = box_right
        ds["bottom"] = box_bottom
              
        insert_distance(ds)

            
    #didn't detect anything. 
    if len(detections) is None:
        print("Nothing detected")

        ds = {}
        ds["object_name"] = "Nothing"
        ds["frame_id"] = frame_number 
        ds["distance"] = 9999 
        ds["left"] = 0 
        ds["top"] = 0
        ds["right"] = 0
        ds["bottom"] = 0
            
        insert_distance(ds)


    #draw on picture

    cv2.rectangle(color_image,(m1_left,m1_top),(m1_right,m1_bottom),(255,255,255), m1_border)
    cv2.rectangle(color_image,(m2_left,m2_top),(m2_right,m2_bottom),(255,255,255), m2_border)
    cv2.rectangle(color_image,(m3_left,m3_top),(m3_right,m3_bottom),(255,255,255), m3_border)
    cv2.rectangle(color_image,(m4_left,m4_top),(m4_right,m4_bottom),(255,255,255), m4_border)
    cv2.rectangle(color_image,(m5_left,m5_top),(m5_right,m5_bottom),(255,255,255), m5_border)


    cv2.putText(color_image,
            	"{:.0f} FPS".format(net.GetNetworkFPS()),
            	(int(width*0.8), int(height*0.1)),
            	cv2.FONT_HERSHEY_SIMPLEX,1,
            	(0,255,255),2,cv2.LINE_AA)
    
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    overlay_image = cv2.addWeighted(color_image, 0.5, colorized_depth, 0.5, 0)


    return cv2.imencode('.jpg', overlay_image)[1].tobytes()
    #return cv2.imencode('.jpg', color_img)[1].tobytes()


@app.route('/distances')
def distance_stream():
    distances = get_distances()
    return jsonify(distances)

@app.route('/averages')
def averages_stream():
    averages = get_averages()
    return jsonify(averages)


@app.route('/get_navigation_suggestion')
def get_navigation_suggestion():
    av, dss = get_most_recent_frame_info()
    
    print(av)
    print(dss)

    m1_chicken_coverage = m2_chicken_coverage = m3_chicken_coverage = m4_chicken_coverage = m5_chicken_coverage = 0
    m1_human_coverage = m2_human_coverage = m3_human_coverage = m4_human_coverage = m5_human_coverage = 0

    global m1_left, m2_left, m3_left, m4_left, m5_left
    global m1_right, m2_right, m3_right, m4_right, m5_right
    global m1_top, m2_top, m3_top, m4_top, m5_top
    global m1_bottom, m2_bottom, m3_bottom, m4_bottom, m5_bottom


    for ds in dss:
        #for each detection, calculate bounding box overlaps with M1/2/3/4/5
        #coverage ranges from 0 to 1
        if ds["object_name"] == "CHICKEN":
            chicken_array = [ds["left"], ds["top"], ds["right"], ds["bottom"]]
            m1_chicken_coverage = bb_iou ([m1_left, m1_top, m1_right, m1_bottom], chicken_array, m1_chicken_coverage )
            m2_chicken_coverage = bb_iou ([m2_left, m2_top, m2_right, m2_bottom], chicken_array, m2_chicken_coverage )
            m3_chicken_coverage = bb_iou ([m3_left, m3_top, m3_right, m3_bottom], chicken_array, m3_chicken_coverage )
            m4_chicken_coverage = bb_iou ([m4_left, m4_top, m4_right, m4_bottom], chicken_array, m4_chicken_coverage )
            m5_chicken_coverage = bb_iou ([m5_left, m5_top, m5_right, m5_bottom], chicken_array, m5_chicken_coverage )

        if ds["object_name"] == "human":
            human_array = [ds["left"], ds["top"], ds["right"], ds["bottom"]]
            print(human_array)
            print([m5_left, m5_top, m5_right, m5_bottom])
            m1_human_coverage = bb_iou ([m1_left, m1_top, m1_right, m1_bottom], human_array, m1_human_coverage )
            m2_human_coverage = bb_iou ([m2_left, m2_top, m2_right, m2_bottom], human_array, m2_human_coverage )
            m3_human_coverage = bb_iou ([m3_left, m3_top, m3_right, m3_bottom], human_array, m3_human_coverage )
            m4_human_coverage = bb_iou ([m4_left, m4_top, m4_right, m4_bottom], human_array, m4_human_coverage )
            m5_human_coverage = bb_iou ([m5_left, m5_top, m5_right, m5_bottom], human_array, m5_human_coverage )

    chicken_coverage = [m1_chicken_coverage, m2_chicken_coverage, m3_chicken_coverage, m4_chicken_coverage, m5_chicken_coverage]
    human_coverage = [m1_human_coverage, m2_human_coverage, m3_human_coverage, m4_human_coverage, m5_human_coverage]
    
    #priorities... chickens, humans, collision
    max_chicken_coverage = max(chicken_coverage)
    max_human_coverage = max(human_coverage)

    if (max_chicken_coverage > 0):
        max_chicken_index = chicken_coverage.index(max_chicken_coverage)
        #0,1,2,3,4
        if max_chicken_index < 2:
            return "Left"
        elif max_chicken_index > 2:
            return "Right"
        elif max_chicken_index == 2:
            if av["m3_avg_close_depth"] < 500:
                return "Chicken"

    elif (max_human_coverage > 0):
        print(human_coverage)
        print(max_human_coverage)
        max_human_index = human_coverage.index(max_human_coverage)
        print(max_human_index)
        #0,1,2,3,4
        if max_human_index < 2:
            return "Left"
        elif max_human_index > 2:
            return "Right"
        elif max_human_index == 2:
            if av["m3_avg_close_depth"] < 500:
                return "Human"
  
    
    close_avgs = [ av["m1_avg_close_depth"], av["m2_avg_close_depth"], av["m3_avg_close_depth"], av["m4_avg_close_depth"], av["m5_avg_close_depth"] ]
    max_avg_close_depth = max(close_avgs)
    max_depth_index = close_avgs.index(max_avg_close_depth)

    if max_depth_index < 2:
        return "Left"
    elif max_depth_index > 2:
        return "Right"
    elif max_depth_index == 2:
        return "Forward"
        
    avgs = [ av["m1_avg_depth"], av["m2_avg_depth"], av["m3_avg_depth"], av["m4_avg_depth"], av["m5_avg_depth"] ]
    max_avg_depth = max(avgs)
    max_depth_index = avgs.index(max_avg_depth)

    if max_depth_index < 2:
        return "Left"
    elif max_depth_index > 2:
        return "Right"
    elif max_depth_index == 2:
        return "Forward"
        

    return "Back"


#[67, 76, 421, 619]
#[512, 160, 640, 320]

#adapted from pyimagesearch - but only care about the navigation 'M' boxes, not the 'union'
#and passes in current overlaps, 
def bb_iou(boxA, boxB, currentMax):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    ionb = interArea / float(boxAArea)

    return ionb if ionb > currentMax else currentMax
    

def camera_stream():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame() 
    color_frame = aligned_frames.get_color_frame()
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_color_frame = rs.colorizer().colorize(aligned_depth_frame)
    depth_color_image = np.asanyarray(depth_color_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    resized_depth_image = cv2.resize(depth_color_image, dsize=(320, 240), interpolation=cv2.INTER_AREA)
    resized_color_image = cv2.resize(color_image, dsize=(320,240), interpolation=cv2.INTER_AREA)

    images = np.hstack((resized_color_image, resized_depth_image))

    return cv2.imencode('.jpg', images)[1].tobytes()


def gen_frame():
    """Video streaming generator function."""
    while True:
        
        #frame = camera_stream()
        frame = inference_stream()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


@app.route('/video_feed')
def video_feed():
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", use_reloader=False)
