import cv2 as cv
from gtts import gTTS
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pyttsx3
import subprocess
import sys
import threading

from google.cloud import automl_v1beta1
from imageai import Detection


DEFAULT_THRESHOLD = 40

def compute_mhi(video_src):
    cam = cv.VideoCapture(video_src)
    if not cam.isOpened():
        print("could not open video_src " + str(video_src) + " !\n")

    cam.set(cv.CAP_PROP_POS_AVI_RATIO,1)
    MHI_DURATION = cam.get(cv.CAP_PROP_POS_MSEC) * 5

    cam = cv.VideoCapture(video_src)
    if not cam.isOpened():
        print("could not open video_src " + str(video_src) + " !\n")

    ret, frame = cam.read()
    if ret == False:
        print("could not read from " + str(video_src) + " !\n")

    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)

    while cam.isOpened():
        ret, frame = cam.read()
        if ret == False:
            break
        frame_diff = cv.absdiff(frame, prev_frame)
        gray_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
        ret, motion_mask = cv.threshold(gray_diff, DEFAULT_THRESHOLD, 1,        
                                        cv.THRESH_BINARY)
        timestamp = cv.getTickCount() / cv.getTickFrequency() * 1000
        cv.motempl.updateMotionHistory(motion_mask, motion_history, timestamp, 
                                       MHI_DURATION)
        vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / \
                                MHI_DURATION, 0, 1) * 255)
        prev_frame = frame.copy()
    cam.release()
    
    return vis

def generate_mhi_picture(video_src):
    mhi = compute_mhi(video_src)

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.matshow(mhi, cmap=plt.cm.gray)
    plt.savefig("mhi.png")

def get_prediction(content, project_id, model_id):
  prediction_client = automl_v1beta1.PredictionServiceClient()

  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'image': {'image_bytes': content }}
  params = {}
  request = prediction_client.predict(name=name, payload=payload, params=params)
  return request  # waits till request is returned


def detect_activities(video_src, objects):
    print("Started generating the MHI...")
    generate_mhi_picture(video_src)
    print("Finished generating the MHI! Started predicting...")
    
    # Call the activity recognition module.
    file_path = "mhi.png"
    project_id = "har-rgb"
    model_id = "ICN8870376055897672796"

    with open(file_path, 'rb') as ff:
        content = ff.read()
    try:
        prediction = get_prediction(content, project_id,  model_id)
        activity = prediction.payload[0].display_name
    except Exception as e:
        print(e)
        activity = "generic activity"
        
    if activity == "drink from a bottle" and "cup" in objects:
        activity = "drink from a mug"
    elif activity == "drink from a mug" and "bottle" in objects:
        activity = "drink from a bottle"

    print("Finished predicting! The prediction result is:")
    print(activity)
    
    # Text to speech: activity.
    myobj = gTTS(text="Detected activity: " + activity, lang='en', slow=False)
    myobj.save("out.mp3")
    os.system("mpg321 out.mp3")


def speak(msg):
    myobj = gTTS(text=msg, lang='en', slow=False)
    myobj.save("out2.mp3")
    os.system("mpg321 out2.mp3")


def detect_objects(video_src):
    yolo = Detection.ObjectDetection()
    yolo.setModelTypeAsYOLOv3()
    yolo.setModelPath("./yolo.h5")
    yolo.loadModel()
    
    cam = cv.VideoCapture(video_src)
    if not cam.isOpened():
        print("could not open video_src " + str(video_src) + " !\n")

    cam.set(cv.CAP_PROP_POS_AVI_RATIO,1)
    MHI_DURATION = cam.get(cv.CAP_PROP_POS_MSEC)

    cam = cv.VideoCapture(video_src)
    if not cam.isOpened():
        print("could not open video_src " + str(video_src) + " !\n")

    ret, img = cam.read()
    if ret == False:
        print("could not read from " + str(video_src) + " !\n")

    i = 0
    while cam.isOpened():
        ret, frame = cam.read()
        i += 1
        if ret == False:
            break
        if i % 13 != 0:   # 26 fps.
            continue

        img, preds = yolo.detectCustomObjectsFromImage(input_image=img, 
                      custom_objects=None, input_type="array",
                      output_type="array",
                      minimum_percentage_probability=70,
                      display_percentage_probability=False,
                      display_object_name=True)
        print(preds)
        for pred in preds:
            if pred['name'] not in objects:
                objects.append(pred['name'])


def main():
    video_src = sys.argv[1]

    # Call the object detector.
    t1 = threading.Thread(target=speak, args=(
        "Started analyzing the objects in the environment..." +
        "Please be patient, this should take a couple of seconds.", ))
    t2 = threading.Thread(target=detect_objects, args=(video_src, ))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(objects)

    # Call the activity detector.
    speak("Detected objects: " + " and ".join(objects))
    speak("Started analyzing the performed activity...")
    detect_activities(video_src, objects)



if __name__ == "__main__":
    objects = []
    main()
