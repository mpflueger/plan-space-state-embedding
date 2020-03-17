import numpy as np
import os

from gym import utils
from gym.envs.mujoco import mujoco_env

# Grabbed from gym and modified
class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, swingup=True, su_reward_v=4):
        utils.EzPickle.__init__(**locals())

        self.swingup = swingup
        self.su_reward_v = su_reward_v

        # Find xml path
        xml_path = os.path.join(os.getcwd(), 'cartpole_swingup.xml')

        mujoco_env.MujocoEnv.__init__(self, xml_path, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

        if self.swingup:
            # Need to have pole theta in -pi to pi
            theta = ((ob[1] + 3.1415) % 6.283) - 3.1415
            #reward_dist = - np.abs(((ob[1] + 3.1415) % 6.283) - 3.1415)
            if self.su_reward_v == 3:
                target_bonus = 0
                if np.abs(ob[1] < 0.1):
                    target_bonus = 1
                #reward_dist = - np.abs(((ob[1] + 3.1415) % 6.283) - 3.1415)
                reward_dist = - np.abs(theta)
                reward_ctrl = -0.1 * np.square(a).sum()
                reward = reward_dist + reward_ctrl + target_bonus
                done = False
            elif self.su_reward_v == 4:
                target_bonus = 0
                if np.abs(theta < 0.1):
                    target_bonus = 1
                #reward_dist = - np.abs(((ob[1] + 3.1415) % 6.283) - 3.1415)
                reward_dist = - np.abs(theta)
                reward_ctrl = -0.1 * np.square(a).sum()
                reward = reward_dist + reward_ctrl + target_bonus
                done = False
            elif self.su_reward_v == 5:
                reward_dist = - np.abs(theta)
                reward_ctrl = -0.1 * np.square(a).sum()
                reward = reward_dist + reward_ctrl
                done = False

                # If upright with low velocity end the episode and give reward
                speed = np.abs(ob[2])
                spin = np.abs(ob[3])
                if np.abs(theta) < 0.1 and speed < 0.1 and spin < 0.1:
                    reward += 100
                    done = True
            else:
                raise NotImplementedError(
                    "Unknown swingup reward version su_reward_v={}".format(
                        su_reward_v))

        else:
            reward = 1.0
            notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
            done = not notdone

        # In case _get_obs is overridded by the embedded class, we
        # need to call it here 
        ob = self._get_obs()
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01)

        # For a swingup start the pole anywhere, with a little more velocity
        if self.swingup:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=[-0.1, -3.14], high=[0.1, 3.14])
            qvel = self.init_qvel + self.np_random.uniform(
                size=self.model.nv, low=[-0.1, -0.5], high=[0.1, 0.5])

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

