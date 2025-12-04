import cv2
import gin
import gymnasium
from vizdoom import gymnasium_wrapper
from gymnasium import spaces
import numpy as np
import os
import tasks.abc_task
import time
from takecover_variants.doom_take_cover import DoomTakeCoverEnv

class GymTask(tasks.abc_task.BaseTask):
    """OpenAI gym tasks."""

    def __init__(self):
        self._env = None
        self._render = False
        self._logger = None

    def create_task(self, **kwargs):
        raise NotImplementedError()
    
    def seed(self, seed):
        if isinstance(self, TakeCoverTask):
            self._env.game.set_seed(seed)
        else:
            self._env.seed(seed)

    def reset(self):
        #print(self._env)
        result = self._env.reset()
        print("DEBUG reset result:", result)
        # Gymnasium: (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return obs,info

        # Classic Gym: obs only
        return result
    
    def step(self, action, evaluate):
        result = self._env._step(action)

        # Gymnasium: (obs, reward, terminated, truncated, info)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated,info = result
            done = terminated or truncated
            return obs, reward, done

        # Classic Gym: (obs, reward, done, info)
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, terminated, truncated = result
            done = terminated or truncated
            return obs, reward, done

        # Something went wrong
        raise ValueError(f"Unexpected env.step() return: {result}")
    
    def close(self):
        self._env.close()

    def _process_reward(self, reward, done, evaluate):
        return reward
    
    def _overwrite_terminate_flag(self, reward, done, step_cnt, evaluate):
        return done
    
    def _show_gui(self):
        if hasattr(self._env, 'render'):
            self._env.render()

    def roll_out(self, solution, evaluate):
        self._logger.info(evaluate)#roll_out info
        ob= self.reset()
        # Extract the image the policy expects
                
        ob = self._process_observation(ob)
        if hasattr(solution, 'reset'):
            solution.reset()

        start_time = time.time()

        rewards = []
        done = False
        step_cnt = 0
        while not done:
            action = solution.get_output(inputs=ob, update_filter=not evaluate)
            action = self._process_action(action)
            ob, reward, done = self.step(action, evaluate)
            #print("at step:"+ob)

            ob = self._process_observation(ob)

            if self._render:
                self._show_gui()

            step_cnt += 1
            done = self._overwrite_terminate_flag(reward, done, step_cnt, evaluate)
            step_reward = self._process_reward(reward, done, evaluate)
            rewards.append(step_reward)

        time_cost = time.time() - start_time
        actual_reward = np.sum(rewards)
        if hasattr(self, '_logger') and self._logger is not None:
            self._logger.info(
                'Roll-out time={0:.2f}s, steps={1}, reward={2:.2f}'.format(
                    time_cost, step_cnt, actual_reward))

        return actual_reward


@gin.configurable
class TakeCoverTask(GymTask):
    """VizDoom take cover task."""

    def __init__(self):
        super(TakeCoverTask, self).__init__()
        self._float_text_env = False
        self._text_img_path = 'takecover_variants/attention_agent.png'

    def create_task(self, **kwargs):
        if 'render' in kwargs:
            self._render = kwargs['render']
        if 'logger' in kwargs:
            self._logger = kwargs['logger']
        modification = 'original'
        if 'modification' in kwargs:
            modification = kwargs['modification']
            if modification == 'text':
                self._float_text_env = True
        self._logger.info('modification: {}'.format(modification))
        print("modification: " + modification)
        self._env = DoomTakeCoverEnv(modification)
        return self

    def _process_observation(self, observation):
        if not self._float_text_env:
            return observation
        img = cv2.imread(self._text_img_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        full_color_patch = np.ones([h, w], dtype=np.uint8) * 255
        zero_patch = np.zeros([h, w], dtype=np.uint8)
        x = 150
        y = 30
        mask = (img == 0)
        observation[y:(y+h), x:(x+w), 0][mask] = zero_patch[mask]
        observation[y:(y+h), x:(x+w), 1][mask] = zero_patch[mask]
        observation[y:(y+h), x:(x+w), 2][mask] = full_color_patch[mask]
        observation[y:(y+h), x:(x+w), 0][~mask] = zero_patch[~mask]
        observation[y:(y+h), x:(x+w), 1][~mask] = full_color_patch[~mask]
        observation[y:(y+h), x:(x+w), 2][~mask] = full_color_patch[~mask]
        return observation

    def _process_action(self, action):
        # Follow the code in world models.
        action_to_apply = [0] * 43
        threshold = 0.3333
        if action > threshold:
            action_to_apply[10] = 1
        if action < -threshold:
            action_to_apply[11] = 1
        return action_to_apply

    def set_video_dir(self, video_dir):
        from gymnasium.wrappers import Monitor
        self._env = Monitor(
            env=self._env,
            directory=video_dir,
            video_callable=lambda x: True
        )
