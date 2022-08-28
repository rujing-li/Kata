#extracting landmarks from a video, then converting the video into a gif with visualization
#save the keypoints as a json file so it can be played again

#imports
import cv2
import mediapipe as mp
import numpy as np
#import imageio
import imageio.v2 as imageio
import pickle       #to open output/.gif
import json
import os

#utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#define video here
video = 'karate-girl.mp4'
video_path = 'videos/' + video

#video feed
cap = cv2.VideoCapture(video_path)

#variables and counters
index = 0
temp_path = 'temp/'
output_path = 'output/'
annotated_frames = []
keypoints = []
current_frame = 0

#create path if it doesnt exist
if not os.path.exists('imgdata'):
    os.makedirs('imgdata')

#setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignore empty frame")
            break

        #detect stuff and render
        #recolour image to RGB
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #make detection
        results = pose.process(image)

        #extraction done here
        #adding keypoints from video to build landmarks
        if results.pose_landmarks is not None:
	        annotated_pose_landmarks = {str(j): [lndmk.x, lndmk.y, lndmk.z] for j, lndmk in enumerate(results.pose_landmarks.landmark)}
	        keypoints.append(annotated_pose_landmarks)

        #recolour image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )

        #building annotated frames
        cv2.imwrite(temp_path + str(index) + '.png', image)
        annotated_frames.append(temp_path + str(index) + '.png')
        index += 1

        # while(True):
        #     ret, frame = cap.read()
        #     cv2.imwrite('./imgdata/frame' + str(current_frame) + '.jpg', frame)
        #     current_frame += 1

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#join frames to create and save a gif
def convert_frames_to_gif(frames, output_gif):
	images = []

    #build gif from frames
	for frame in frames:
		images.append(imageio.imread(frame))

    #store gif and create output path
	imageio.mimsave(output_path + output_gif + '.gif', images)

#converts keypoints dictionary to a json file
with open(output_path + video + '-keypoints.json', 'w') as fp:
    json.dump(keypoints, fp)

#convert_frames_to_gif(annotated_frames, video)