import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import time

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

async def send_data():

    uri = "ws://192.168.1.10:8765"

    async with websockets.connect(uri) as websocket:

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            blink_rate = 0

            if results.multi_face_landmarks:
                blink_rate = 1  # simplified placeholder

            payload = {
                "timestamp": time.time(),
                "blink_rate": blink_rate
            }

            await websocket.send(json.dumps(payload))

            await asyncio.sleep(1)

asyncio.run(send_data())