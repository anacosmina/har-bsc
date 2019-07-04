import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2

DEFAULT_THRESHOLD = 30

def compute_mhi(video_src):
    cam = cv.VideoCapture(video_src)
    if not cam.isOpened():
        print("could not open video_src " + str(video_src) + " !\n")

    cam.set(cv.CAP_PROP_POS_AVI_RATIO,1)
    MHI_DURATION = cam.get(cv.CAP_PROP_POS_MSEC)

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

def generate_mhi_picture():
    video_src = sys.argv[1]
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
  request = prediction_client.predict(name, payload, params)
  return request  # waits till request is returned


def main():
    print("Started generating the MHI...")
    generate_mhi_picture()
    print("Finished generating the MHI! Started predicting...")
    
    # Call AutoML's API.
    file_path = "mhi.png"
    project_id = "har-rgb"
    model_id = "ICN8870376055897672796"

    with open(file_path, 'rb') as ff:
        content = ff.read()

    prediction = get_prediction(content, project_id,  model_id)
    print("Finished predicting! The prediction result is:")
    print(prediction)


if __name__ == "__main__":
    main()
