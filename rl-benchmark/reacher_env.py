import akro
import mujoco_py as mj
import numpy as np
import os

from gym import utils
import gym.envs.mujoco.reacher as gym_reacher
from gym.envs.mujoco import mujoco_env

from garage.envs.base import GarageEnv
from garage.envs.base import Step


# Grabbed from gym and modified
class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, r_version=2, obs_theta=False, obs_trig=True, obs_target=True):
        utils.EzPickle.__init__(**locals())

        self.r_version = r_version

        self.obs_theta = obs_theta
        self.obs_trig = obs_trig
        self.obs_target = obs_target

        # Find xml path
        xml_path = os.path.join(os.getcwd(), 'reacher_limited.xml')

        mujoco_env.MujocoEnv.__init__(self, xml_path, 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        if self.r_version == 1:
            reward = reward_dist + reward_ctrl
        elif self.r_version == 2:
            reward = reward_dist + (0.1 * reward_ctrl)
        else:
            raise NotImplementedError(
                "ReacherEnv: Unknown reward version: ", self.r_version)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        obs_list = []
        if self.obs_theta:
            obs_list += [theta]
        if self.obs_trig:
            obs_list += [np.cos(theta)]
            obs_list += [np.sin(theta)]
        if self.obs_target:
            obs_list += [self.sim.data.qpos.flat[2:]]
        obs_list += [self.sim.data.qvel.flat[:2]]
        if self.obs_target:
            obs_list += [self.get_body_com("fingertip")
                         - self.get_body_com("target")]
        return np.concatenate(obs_list)

