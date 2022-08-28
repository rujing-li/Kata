# main app for Kata

# imports
from flask import Flask, render_template, request, Response
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import pickle
import imageio.v2 as imageio
app = Flask(__name__)

# utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# initialize variables
tutorial_name = 'karate-girl-2.mp4-keypoints'

# citing: from HTN's smart dance project


def connect_points(points, translation_factors, image, image_shape, scale):
    h, w = image_shape
    points_connect_dict = {
        1: [2, 0],
        2: [3],
        3: [7],
        4: [0, 5],
        5: [6],
        6: [8],
        9: [10],
        11: [13],
        12: [11, 14],
        13: [15],
        14: [16],
        15: [21],
        16: [20, 14],
        17: [15],
        18: [20, 16],
        19: [17],
        20: [16],
        22: [16],
        23: [11, 25],
        24: [23, 12],
        25: [27],
        26: [24, 28],
        27: [31, 29],
        28: [30, 32],
        29: [31],
        30: [32],
        32: [28],
    }
    for p in points_connect_dict:
        curr_point = points[str(p)][0:2] * np.array([w, h]) - \
            np.array(list(translation_factors))

        for endpoint in points_connect_dict[p]:
            endpoint = points[str(endpoint)][0:2] * np.array([w, h]) - \
                np.array(list(translation_factors))

            cv2.line(image, (round(curr_point[0] * scale), round(curr_point[1] * scale)),
                     (round(endpoint[0] * scale), round(endpoint[1] * scale)), (0, 0, 255), thickness=10)

    return image

# citing: from HTN's smart dance project


def get_translation_factor(gt, person, h, w):

    x_gt, y_gt = gt['7'][0]*w, gt['7'][1]*h
    x_person, y_person = person[7][0]*w, person[7][1]*h

    if x_person >= x_gt:
        return x_person - x_gt, y_person - y_gt
    elif x_person <= x_gt:
        return x_gt - x_person, y_gt - y_person


def l2_norm(actual_landmarks, user_landmarks):
    return np.linalg.norm(actual_landmarks - user_landmarks)

# compare landmarks between the user's pose (from webcam) and the actual video (from self defense tutorial)


def compare_keypoints(actual_keypoints, user_keypoints, w, h, translation_factors):

    # initialize
    actual_keypoints_array = []
    user_keypoints_array = []

    # out of x, y, z and visibility data,
    # metric compares using x and y data
    for i in range(len(actual_keypoints)):
        actual_keypoints_array.append(np.array(actual_keypoints[str(i)])[
                                      0:2] * np.array([w, h]))
                                      #- np.array(list(translation_factors)))
        user_keypoints_array.append(np.array(user_keypoints[i])[
                                0:2] * np.array([w, h]))

    # creating a single array
    actual_keypoints_array = np.vstack(actual_keypoints_array)
    user_keypoints_array = np.vstack(user_keypoints_array)

    return l2_norm(actual_keypoints_array, user_keypoints_array)


def put_text(image, text, h, w):
    image = cv2.putText(img=image, org=(w - 700, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 0), text=text,
                        thickness=3)
    return image

# create tutorial landmarks for user to follow along


def create_tutorial_landmark():

    ######      VISUALIZING ALL THE LANDMARKS FROM THE TUTORIAL VIDEO       #####

    with open('tutorials/' + tutorial_name + '.json') as f:
        tutorial_data = json.load(f)

    # min counter
    counter = 0
    cap = cv2.VideoCapture(0)

    # min, max, and update counters
    counter_update = 1
    max_counter = 161

    while True and counter_update <= (max_counter-1):

        counter_update += 1
        ret, image = cap.read()
        if not ret:
            pass
        else:
            # detect stuff and render
            # recolour image to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            h, w, _ = image.shape

            # visualize video
            image = cv2.putText(img=image, org=(w//2, h//2),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 0),
                                text=str((max_counter - counter_update)//40), thickness=2)

            # convert back to bgr
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--image\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

    ######      VISUALIZING THE USER FROM THEIR CAMERA       #####

    avg = []

    while True:
        if counter == len(tutorial_data) - 1:
            counter = 0
        else:
            counter += 1

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                # recolour image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                copy_image = image
                image.flags.writeable = False

                # make detection
                results = pose.process(image)
                pose_landmarks = results.pose_landmarks

                if pose_landmarks is not None:

                    scale_t = 1.0
                    h, w, _ = image.shape
                    pose_landmarks_str_keys = {str(i): [lndmk.x, lndmk.y, lndmk.z]
                                        for i, lndmk in enumerate(pose_landmarks.landmark)}
                    pose_landmarks = {i: [lndmk.x, lndmk.y, lndmk.z]
                                        for i, lndmk in enumerate(pose_landmarks.landmark)}
                    tutorial_data[counter] = {i: [scale_t*keypoint for keypoint in tutorial_data[counter][i]] for i in tutorial_data[counter]}

                    x_t, y_t = get_translation_factor(tutorial_data[counter], pose_landmarks, h, w)

                    # recolour image back to BGR
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    #render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                    ######      COMPARING VISUALIZATION OF TUTORIAL LANDMARKS TO USER LANDMARKS      #####

                    comparisons = connect_points(tutorial_data[counter], (x_t, y_t), copy_image, (h, w), scale=1.0)
                    comparisons = cv2.cvtColor(comparisons, cv2.COLOR_RGB2BGR)
                    comparisons = connect_points(pose_landmarks_str_keys, (0, 0), comparisons, (h, w), scale=1.0)
                    points = compare_keypoints(tutorial_data[counter], pose_landmarks, w, h, (x_t, y_t))

                    if len(avg) == 20:
                        avg.pop(0)
                        avg.append(points)
                    else:
                        avg.append(points)

                    comparisons = put_text(
                            comparisons, "Score :" + str(round(sum(avg)/len(avg), 2)), h, w)
                    ret, buffer = cv2.imencode('.jpg', comparisons)
                    image = buffer.tobytes()
                    yield (b'--image\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start')
def start():
    return render_template('start.html')


@app.route('/videos/<id>')
def video(id):
    return render_template('video.html', id=id)


@app.route('/trainings/<id>')
def training(id):
    # return render_template('training.html', id = id)
    return Response(create_tutorial_landmark(), mimetype='multipart/x-mixed-replace; boundary=image')


@app.route('/info')
def info():
    return render_template('info.html')


if __name__ == '__main__':
    app.run(debug=True)
