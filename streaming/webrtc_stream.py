from aiortc import RTCPeerConnection, VideoStreamTrack
import cv2

class CameraStream(VideoStreamTrack):

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)

    async def recv(self):
        ret, frame = self.cap.read()
        return frame