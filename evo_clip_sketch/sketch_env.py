import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
import time

from PIL import Image
import cv2
from gym.spaces import Box

import clip
import torch

class SketchEnv:
    def __init__(self):

        self.episode = 0
        self.last_reward = 0.
        self.last_rewards = []
        self.reward = None
        self.max_reward = -1
        self.past_points = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = None, None

        self.target_dir = None
        self.target_shape = None
        self.max_steps = None
        self.target_words = None
        self.target_thickness = None
        self.factor_height, self.factor_width = None, None

    def reset(self):
        
        self.episode = 0
        self.sketch = np.zeros(
                            shape=self.target_shape, 
                            dtype='uint8'
                      ) + 255
        self.last_points = []
     
        return self.observation()

    def _to_emb(self, np_img):

        img = Image.fromarray(np_img)
        img = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.clip_model.encode_image(img)
        vec = torch.squeeze(vec).detach().unsqueeze(0).numpy()

        return vec

    def clip_reward(self):

        img = Image.fromarray(self.sketch)
        image = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        text = clip.tokenize(self.target_words).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
            
            return similarity[0][0]

    def observation(self):

        return None, self.sketch

    def step(self,action):

        self.episode += 1
               
        if len(self.last_points) == 0:
            x_abs, y_abs = [np.clip(action[i] + .5, 0., 1.) for i in range(self.action_dim)]
            point_abs = (x_abs * self.factor_width, int(y_abs * self.factor_height))
            self.last_points.append(point_abs)
        else:
            last_point = self.last_points[-1]
            x_change, y_change = [np.clip(action[i], -1., 1.) for i in range(self.action_dim)]
            point_change = (int(x_change * self.factor_width), int(y_change * self.factor_height))
            relative_point = (last_point[0] + point_change[0], last_point[1] - point_change[1])
            relative_point = [
                                np.clip(relative_point[i], 0., self.sketch.shape[i]) 
                                for i in range(self.action_dim)
                             ] #could check before this if values become to large or small

            self.last_points.append(relative_point)
        
        self.reward = 0.
        terminal = False if self.episode < self.max_steps else True
        if terminal:

            blank_sketch = np.zeros(
                                shape=self.sketch.shape, 
                                dtype='uint8'
                           ) + 255
            closed = True
            color = (0, 0, 0) #black
            points = np.array(
                            self.last_points, 
                            dtype=np.int32
                     )
            self.sketch = cv2.polylines(
                                blank_sketch, 
                                [points],
                                closed, 
                                color, 
                                self.target_thickness
                            )
                            
            self.reward = self.clip_reward()
            self._save_sketch()
        
        return self.observation(), self.reward, terminal, None

    def render(self):
      
        cv2.imshow(
                'sketch for "{}"'.format(self.target_words[0]),
                self.sketch
        )
        cv2.waitKey(1)
    
    def _init_all(
            self,
            config,
    ):

        self.max_steps = config['max_steps']
        self.target_words = [config['target_words']]
        self.target_shape = config['target_shape']
        self.target_thickness = config['target_thickness']
        point_reduction_factor = config['point_reduction_factor']
        height, width, _ = self.target_shape
        self.factor_height, self.factor_width = height // point_reduction_factor, width // point_reduction_factor
        self._set_target_dir(config['target_dir'])
        self.action_dim = config['num_actions']
        self.action_space = Box( # values should be between -1 and 1
                                np.array([-1.] * self.action_dim), 
                                np.array([1.] * self.action_dim)
                            )
        self.clip_model, \
        self.clip_preprocess = clip.load(
                                        config['clip_model'], 
                                        device=self.device
                               )
      
        print("got config: {}".format(config))

    def _set_target_dir(self, dir):

        if dir is not None:

            concat_target_words = self._get_concatenated_word()
            new_dir = "{}{}/".format(dir, concat_target_words)
            try:
                shutil.rmtree(new_dir)
            except:
                pass
            Path(new_dir).mkdir(exist_ok=True, parents=True)
            self.target_dir = new_dir

    def _save_sketch(self):
        
        if self.reward > self.max_reward and self.target_dir is not None:
            unix = time.mktime(datetime.now().timetuple())
            path = '{}_{}_{}.png'.format(self.target_dir, self.reward, str(unix))
            cv2.imwrite(path, self.sketch)
            self.max_reward = self.reward

    def _get_concatenated_word(self):

        concat_target_words = ""
        for word in self.target_words:
            splitted = word.split(' ')
            joined = "-".join(splitted)
            concat_target_words = "-" + concat_target_words if len(concat_target_words) != 0 else concat_target_words
            concat_target_words = concat_target_words + joined

        return concat_target_words

