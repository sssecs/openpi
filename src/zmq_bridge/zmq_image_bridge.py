import zmq
import json
import numpy as np
import cv2


class ZMQImageReceiver:
    """
    A ZeroMQ image subscriber that receives ROS2-published sensor_msgs/Image data
    (sent by your C++ image_zmq_bridge node) and converts it to a (224, 224, 3) np.uint8 array.
    """

    def __init__(self, address: str = "tcp://localhost:5555", resize=(224, 224)):
        """
        Initialize the ZMQ subscriber.
        :param address: The ZMQ address to connect to, e.g. "tcp://localhost:5555"
        :param resize: Target resize dimensions (width, height)
        """
        self.address = address
        self.resize = resize

        # Initialize ZMQ
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(self.address)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[ZMQImageReceiver] Connected to {self.address}")

    def receive_once(self) -> np.ndarray:
        """
        Receive a single image and return it as a resized numpy array of shape (H, W, 3).
        Returns None if decoding fails.
        """
        try:
            # Receive metadata
            meta = json.loads(self.sub.recv().decode())
            # Receive raw bytes
            img_bytes = self.sub.recv()

            width, height = meta["width"], meta["height"]
            encoding = meta["encoding"]

            # Convert bytes to numpy array
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)

            # Interpret encoding
            if encoding == "rgb8":
                img_np = img_np.reshape((height, width, 3))
            elif encoding == "bgr8":
                img_np = img_np.reshape((height, width, 3))
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            elif encoding == "mono8":
                img_np = img_np.reshape((height, width))
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            else:
                print(f"[ZMQImageReceiver] Unsupported encoding: {encoding}")
                return None

            # Resize and enforce dtype
            resized = cv2.resize(img_np, self.resize, interpolation=cv2.INTER_AREA)
            resized = resized.astype(np.uint8)

            # Guarantee shape
            if resized.shape != (self.resize[1], self.resize[0], 3):
                print(f"[ZMQImageReceiver] Unexpected shape: {resized.shape}")
                return None

            return resized

        except Exception as e:
            print(f"[ZMQImageReceiver] Error receiving image: {e}")
            return None

    def close(self):
        """Close ZMQ resources cleanly."""
        self.sub.close()
        self.ctx.term()
        print("[ZMQImageReceiver] Closed connection")