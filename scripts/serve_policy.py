import logging
import socket
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def main() -> None:
    config_name = "pi0_piper_lora"
    episode_num = "60"
    train_setp = "4999"

    config = _config.get_config(config_name)
    checkpoint_dir = "/mnt/hdd/openpi/checkpoints/" + config_name + "_" + episode_num + "_episode/" + train_setp
    print(checkpoint_dir)
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    example = {
        "observation/state": np.random.rand(7),
        "observation/head_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "task": "do something",
    }
    policy.infer(example)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port="8000",
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
