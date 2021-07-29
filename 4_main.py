import pickle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

with open('pose_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False

        result = holistic.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img.flags.writeable = True

        mp_drawing.draw_landmarks(img, result.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        try:
            pose = result.pose_landmarks.landmark
            pose_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            face = result.face_landmarks.landmark
            face_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            row = pose_row + face_row

            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_probability = model.predict_proba(X)[0]
            print(body_language_class, body_language_probability)

            cv2.rectangle(img, (0, 0), (250, 60), (245, 117, 16), -1)

            cv2.putText(img, 'CLASS',
                        (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, body_language_class.split(' ')[0],
                        (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(img, 'PROB',
                        (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(round(body_language_probability[np.argmax(body_language_probability)],2)),
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass

        cv2.imshow('MediaPipe Feed', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
