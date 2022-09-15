import urllib.request
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from playsound import playsound
from threading import Thread

ALARM_ON = False

def sound_alarm():
	# play an alarm sound
	playsound("Industrial.wav")

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

url='http://192.168.1.4:8080/shot.jpg'

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)
        if EAR < 0.20:
            cv2.putText(img, "Alert", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
            if not ALARM_ON:
                ALARM_ON = True

                t = Thread(target=sound_alarm)
                t.deamon = True
                t.start()
            else:
                ALARM_ON = False

            print("dormir")
        print(EAR)

    cv2.imshow('test', img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
exit(0)

