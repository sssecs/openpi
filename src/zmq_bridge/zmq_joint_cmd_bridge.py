import zmq
import numpy as np
import json
import time


class ZMQJointCmdSender:
    """Send 7-element NumPy arrays via ZMQ PUB socket."""

    def __init__(self, address: str = "tcp://localhost:6000"):
        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(address)
        print(f"[ZMQJointCmdSender] Connected to {address}")

    def send_array(self, arr: np.ndarray):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (7,):
            raise ValueError("Expected a 7-element array.")
        msg = json.dumps(arr.tolist())
        self.pub.send_string(msg)
        print(f"[ZMQJointCmdSender] Sent: {arr}")

    def close(self):
        self.pub.close()
        self.ctx.term()


# Example usage
if __name__ == "__main__":
    sender = ZMQJointCmdSender("tcp://localhost:6000")
    try:
        while True:
            data = np.random.randn(7)
            sender.send_array(data)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        sender.close()
