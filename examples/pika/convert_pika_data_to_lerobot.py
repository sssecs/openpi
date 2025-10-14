import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import shutil
import cv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import json
import numpy as np


REPO_NAME = "sssecs/pika_test"  # Name of the output dataset, also used for the Hugging Face Hub
DATASET_DIR = "/home/markov/data"
PUSH_TO_HUB = True
FPS = 30


def flatten_list(target):
    return [item for sublist in target for item in sublist]


def find_all_hdf5(dataset_dir):
    hdf5_files = []
    for f in os.listdir(dataset_dir):
        hdf5_files.append(os.path.join(os.path.join(dataset_dir, f), "data.hdf5"))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files


def get_all_episode_len(dataset_path_list):
    all_episode_len = []
    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                all_episode_len.append(root['size'][()])
        except Exception as e:
            print(e)
            quit()
    return all_episode_len


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path_list, episode_ids, episode_len):
        super(EpisodicDataset).__init__()
        self.dataset_path_list = dataset_path_list
        self.episode_ids = episode_ids
        self.episode_len = episode_len
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.transformations = None
        self.__getitem__(0)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)  # argmax returns first True index
        start_index = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_index

    def __getitem__(self, index):
        pass

    def get_step(self, episode_id, step):
        start_index = step
        dataset_path = self.dataset_path_list[episode_id]
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/arm/jointStatePosition/master'][()]
            action = root['/arm/jointStatePosition/puppet'][()]

            qpos = torch.from_numpy(qpos[start_index]).float()
            action = torch.from_numpy(action[start_index]).float()


            wrist_image_path = root[f'/camera/color/pikaGripperDepthCamera'][start_index].decode('utf-8')
            wrist_image_path = os.path.join(os.path.dirname(dataset_path), wrist_image_path)
            wrist_image = cv2.imread(wrist_image_path)
            if wrist_image is not None:
                wrist_image = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                wrist_image = cv2.resize(wrist_image, (256, 256))
            else:
                wrist_image = np.zeros((256, 256, 3), dtype=np.uint8)

            fisheye_image_path = root[f'/camera/color/pikaGripperFisheyeCamera'][start_index].decode('utf-8')
            fisheye_image_path = os.path.join(os.path.dirname(dataset_path), fisheye_image_path)
            fisheye_image = cv2.imread(fisheye_image_path)
            if fisheye_image is not None:
                fisheye_image = cv2.cvtColor(fisheye_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                fisheye_image = cv2.resize(fisheye_image, (256, 256))
            else:
                fisheye_image = np.zeros((256, 256, 3), dtype=np.uint8)


            return {
                'state': qpos,
                'action': action,
                'fisheye_image': fisheye_image,
                'wrist_image': wrist_image,
                'episode_index': episode_id,
                'timestamp': start_index,
                'task': "test_task",
            }

    def __len__(self):
        return self.cumulative_len[-1] if len(self.cumulative_len) > 0 else 0


def cleanup_existing_dataset(repo_id):
    """Remove existing dataset if it exists"""
    # Updated cache path for newer LeRobot versions
    cache_dir = os.path.expanduser("~/.cache/huggingface/lerobot")
    dataset_cache_path = os.path.join(cache_dir, repo_id.replace('/', os.sep))
    
    if os.path.exists(dataset_cache_path):
        print(f"Removing existing dataset: {dataset_cache_path}")
        shutil.rmtree(dataset_cache_path)
    else:
        print(f"No existing dataset found at: {dataset_cache_path}")


def create_lerobot_dataset_from_raw(dataset_dir, repo_id):
    """
    Main function to convert raw dataset to LeRobot format
    """

    # Clean up existing dataset first
    cleanup_existing_dataset(repo_id)


    # Step 1: Find all HDF5 files
    dataset_path_list = find_all_hdf5(dataset_dir)
    if not dataset_path_list:
        raise ValueError(f"No HDF5 files found in {dataset_dir}")
    
    # Step 2: Get episode information
    all_episode_len = get_all_episode_len(dataset_path_list)
    episode_ids = list(range(len(dataset_path_list)))
    
    # Step 3: Create episodic dataset
    episodic_dataset = EpisodicDataset(dataset_path_list, episode_ids, all_episode_len)
    
    print(f"Total episodes: {len(dataset_path_list)}")
    print(f"Total frames: {len(episodic_dataset)}")
    
    # Step 4: Determine state and action dimensions from first sample
    sample = episodic_dataset.get_step(0,0)
    state_dim = sample['state'].shape[0]
    action_dim = sample['action'].shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Step 5: Create LeRobot dataset structure
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="pika", 
        fps=FPS, 
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Step 6: Populate the dataset
    print("Populating LeRobot dataset...")
    
    for i in episode_ids:
        for j in range(all_episode_len[i]):
            sample = episodic_dataset.get_step(i,j)
            frame_data = {
                "image": sample['fisheye_image'],
                "wrist_image": sample['wrist_image'],
                "state": sample['state'],
                "action": sample['action'],
                "task": sample["task"],
            }
            dataset.add_frame(frame_data)
        dataset.save_episode()
    
    print(f"Successfully added {len(episodic_dataset)} frames to LeRobot dataset")
    
    return dataset


def convert_and_upload(dataset_dir, repo_id, push_to_hub=True):
    """
    Complete conversion and upload pipeline
    """
    # Convert to LeRobot format
    dataset = create_lerobot_dataset_from_raw(dataset_dir, repo_id)
    
    if push_to_hub:
        # Push to Hugging Face Hub
        print(f"Pushing dataset to {repo_id}...")
        dataset.push_to_hub()
        print("Dataset uploaded successfully!")
    
    return dataset


def main():
    try:
        # Convert with images
        dataset = convert_and_upload(
            dataset_dir=DATASET_DIR,
            repo_id=REPO_NAME,
            push_to_hub=PUSH_TO_HUB  # Set to False if you don't want to upload
        )
        
        # Test the dataset
        print(f"Dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"State shape: {sample['state'].shape}")
        print(f"Action shape: {sample['action'].shape}")
        if 'image' in sample:
            print(f"Image shape: {sample['image'].shape}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == '__main__':
    main()