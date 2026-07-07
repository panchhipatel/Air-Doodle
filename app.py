import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import mediapipe as mp
st.write("MediaPipe version:", getattr(mp, "__version__", "Unknown"))
st.write("Has solutions:", hasattr(mp, "solutions"))
st.write("MediaPipe location:", mp.__file__)
import time


st.set_page_config(page_title="Air Drawing System", layout="wide")
st.title("🎨 Air Drawing System")


class AirDrawingProcessor(VideoProcessorBase):

    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.5
        )

        self.mp_face = mp.solutions.face_mesh
        self.facemesh = self.mp_face.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1
        )

        self.mp_draw = mp.solutions.drawing_utils

        self.canvas = None

        self.current_color = (0, 0, 255)

        self.prev_x = 0
        self.prev_y = 0

        self.prev_hand_points = None


    def recv(self, frame):

        frame = frame.to_ndarray(format="bgr24")

        frame = cv2.flip(frame, 1)

        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_result = self.hands.process(rgb)
        face_result = self.facemesh.process(rgb)

        cv2.rectangle(frame, (0, 0), (80, 640), (50, 50, 50), -1)

        cv2.rectangle(frame, (10, 10), (70, 100), (255, 0, 0), -1)
        cv2.rectangle(frame, (10, 110), (70, 200), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 210), (70, 300), (0, 0, 255), -1)
        cv2.rectangle(frame, (10, 310), (70, 400), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 410), (70, 500), (0, 0, 0), -1)

        cv2.putText(
            frame,
            "CLEAR",
            (15, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # ---------------- FACE ---------------- #

        if face_result.multi_face_landmarks:

            for face in face_result.multi_face_landmarks:

                eye = face_result.multi_face_landmarks[0].landmark

                h, w, _ = frame.shape

                ex = int(eye[468].x * w)
                ey = int(eye[468].y * h)

                cv2.circle(frame, (ex, ey), 5, (255, 255, 255), -1)

                if ex < 80:

                    if 10 < ey < 100:
                        self.current_color = (255, 0, 0)

                    elif 110 < ey < 200:
                        self.current_color = (0, 255, 0)

                    elif 210 < ey < 300:
                        self.current_color = (0, 0, 255)

                    elif 310 < ey < 400:
                        self.current_color = (0, 0, 0)

                    elif 410 < ey < 500:
                        self.canvas[:] = 0

        # ---------------- HAND ---------------- #

        if hand_result.multi_hand_landmarks:

            for hand in hand_result.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS
                )

                lm = hand.landmark

                h, w, _ = frame.shape

                ix = int(lm[8].x * w)
                iy = int(lm[8].y * h)

                mx = int(lm[12].x * w)
                my = int(lm[12].y * h)

                if lm[8].y < lm[6].y:

                    # Two fingers up -> stop drawing
                    if lm[12].y < lm[10].y:

                        self.prev_x = 0
                        self.prev_y = 0

                    # One finger up -> draw
                    else:

                        if self.prev_x != 0 and self.prev_y != 0:
                            ix = int(self.prev_x * 0.7 + ix * 0.3)
                            iy = int(self.prev_y * 0.7 + iy * 0.3)

                        cv2.circle(
                            frame,
                            (ix, iy),
                            10,
                            self.current_color,
                            cv2.FILLED
                        )

                        if self.prev_x == 0 and self.prev_y == 0:
                            self.prev_x = ix
                            self.prev_y = iy

                        thickness = 25 if self.current_color == (0, 0, 0) else 6

                        cv2.line(
                            self.canvas,
                            (self.prev_x, self.prev_y),
                            (ix, iy),
                            self.current_color,
                            thickness
                        )

                        self.prev_x = ix
                        self.prev_y = iy

                else:

                    self.prev_x = 0
                    self.prev_y = 0

        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)

        _, img_inv = cv2.threshold(
            img_gray,
            20,
            255,
            cv2.THRESH_BINARY_INV
        )

        img_inv = cv2.cvtColor(
            img_inv,
            cv2.COLOR_GRAY2BGR
        )

        frame = cv2.bitwise_and(frame, img_inv)
        output = cv2.bitwise_or(frame, self.canvas)

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )

webrtc_streamer(
    key="airdrawing",
    video_processor_factory=AirDrawingProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)