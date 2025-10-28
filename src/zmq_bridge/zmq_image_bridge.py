import zmq
import json
import numpy as np
import cv2


class ZMQImageReceiver:
    """
    A ZeroMQ image subscriber that receives ROS2-published sensor_msgs/Image data
    (sent by your C++ image_zmq_bridge node) and converts it to a (224, 224, 3) np.uint8 array.
    """

    def __init__(self, address: str = "tcp://localhost:5555", resize=(224, 224), encoding="bgr8", width=640, height=480):
        """
        Initialize the ZMQ subscriber.
        :param address: The ZMQ address to connect to, e.g. "tcp://localhost:5555"
        :param resize: Target resize dimensions (self.width, self.height)
        """
        self.address = address
        self.resize = resize
        self.width = width
        self.height = height
        self.encoding = encoding

        # Initialize ZMQ
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.connect(self.address)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[ZMQImageReceiver] Connected to {self.address}")

    def receive_once(self) -> np.ndarray:
        """
        Receive the most recent image (multipart ZMQ message) and return it
        as a resized numpy array of shape (H, W, 3).
        Returns None if decoding or reception fails.
        """
        try:
            # Receive both frames (metadata + image) atomically
            img_bytes = self.sub.recv()
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)

            # Decode according to self.encoding
            if self.encoding == "rgb8":
                img_np = img_np.reshape((self.height, self.width, 3))
            elif self.encoding == "bgr8":
                img_np = img_np.reshape((self.height, self.width, 3))
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            elif self.encoding == "mono8":
                img_np = img_np.reshape((self.height, self.width))
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            else:
                print(f"[ZMQImageReceiver] Unsupported self.encoding: {self.encoding}")
                return None

            # Resize safely
            resized = cv2.resize(img_np, self.resize, interpolation=cv2.INTER_AREA).astype(np.uint8)

            # Validate output shape
            if resized.shape != (self.resize[1], self.resize[0], 3):
                print(f"[ZMQImageReceiver] Unexpected shape: {resized.shape}")
                return None

            return resized

        except zmq.Again:
            # No message available yet (non-blocking mode)
            return None
        except Exception as e:
            print(f"[ZMQImageReceiver] Error receiving image: {e}")
            return None

    def close(self):
        """Close ZMQ resources cleanly."""
        self.sub.close()
        self.ctx.term()
        print("[ZMQImageReceiver] Closed connection")