import os
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import jax
import cv2
import tensorflow_datasets as tfds
import tqdm
import mediapy
import numpy as np
from ros2_octo.octo.model.octo_model import OctoModel

class OctoInferer():
    def __init__(self, language_instruction):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
        self.model = OctoModel.load_pretrained("/home/jonathan/Thesis/octo/umi_checkpoints")

        # sample episode + resize to 256x256 (default third-person cam resolution)
        episode = next(iter(ds))
        steps = list(episode['steps'])
        images = [cv2.resize(np.array(step['observation']['image']), (256, 256)) for step in steps]

        self.window_size = 20

        self.task = self.model.create_tasks(texts=[language_instruction])      

def run_inference(images):
    # input_images = np.stack(images[step:step+self.window_size])[None]
    observation = {
        'image_primary': images,
        'timestep_pad_mask': np.full((1, images.shape[1]), True, dtype=bool)
    }
    
    # this returns *normalized* actions --> we need to unnormalize using the dataset statistics
    actions = model.sample_actions(
        observation, 
        task, 
        unnormalization_statistics=model.dataset_statistics["action"], 
        rng=jax.random.PRNGKey(0)
    )
    actions = actions[0] # remove batch dim

    pred_actions.append(actions)
    final_window_step = step + self.window_size - 1
    true_actions.append(np.concatenate(
        (
            steps[final_window_step]['action'][:3], 
            steps[final_window_step]['action'][3:6], 
            np.array(steps[final_window_step]['action'][7]).astype(np.float32)[None]
        ), axis=-1
    ))