import zmq
import json
import numpy as np


class ZMQJointPosReceiver:
    """
    A ZeroMQ subscriber that receives JointState messages as JSON
    and returns only the joint positions as a NumPy array.
    """

    def __init__(self, address: str = "tcp://localhost:5556"):
        """
        Initialize the ZMQ subscriber.
        :param address: ZMQ endpoint to connect to (default: tcp://localhost:5556)
        """
        self.address = address
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(self.address)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[ZMQJointPosReceiver] Connected to {self.address}")

    def receive_once(self) -> np.ndarray | None:
        """
        Receive one message and return the joint positions as a NumPy array.
        Returns None if the message is invalid or missing 'position'.
        """
        try:
            msg = self.sub.recv()
            data = json.loads(msg.decode("utf-8"))

            if "position" not in data:
                print("[ZMQJointPosReceiver] Warning: no 'position' field in message.")
                return None

            pos = np.array(data["position"], dtype=np.float64)
            return pos

        except Exception as e:
            print(f"[ZMQJointPosReceiver] Error: {e}")
            return None

    def close(self):
        """Cleanly close the ZMQ socket and context."""
        self.sub.close()
        self.ctx.term()
        print("[ZMQJointPosReceiver] Closed connection.")


# --- Example usage ---
if __name__ == "__main__":
    receiver = ZMQJointPosReceiver("tcp://localhost:5556")

    try:
        while True:
            joint_pos = receiver.receive_once()
            if joint_pos is not None:
                print("Joint positions:", joint_pos)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        receiver.close()
