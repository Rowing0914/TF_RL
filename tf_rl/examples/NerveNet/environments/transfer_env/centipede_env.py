#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @brief:
#       The Centipede environments.
#   @author:
#       Tingwu (Wilson) Wang, Aug. 30nd, 2017
# -----------------------------------------------------------------------------

import os

import num2words
import numpy as np
from bs4 import BeautifulSoup as bs
from gym import utils
from gym.envs.mujoco import mujoco_env

import environments.init_path as init_path

DEFAULT_SIZE = 500


class CentipedeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    '''
        @brief:
            In the CentipedeEnv, we define a task that is designed to test the
            power of transfer learning for gated graph neural network.
        @children:
            @CentipedeFourEnv
            @CentipedeEightEnv
            @CentipedeTenEnv
            @CentipedeTwelveEnv
    '''

    def __init__(self, CentipedeLegNum=4, is_crippled=False):

        # get the path of the environments
        if is_crippled:
            xml_name = 'CpCentipede' + self.get_env_num_str(CentipedeLegNum) + \
                       '.xml'
        else:
            xml_name = 'Centipede' + self.get_env_num_str(CentipedeLegNum) + \
                       '.xml'
        self.xml_path = os.path.join(init_path.get_base_dir(),
                                     'environments', 'assets',
                                     xml_name)
        self.xml_path = str(os.path.abspath(self.xml_path))
        self.num_body = int(np.ceil(CentipedeLegNum / 2.0))
        self._control_cost_coeff = .5 * 4 / CentipedeLegNum
        self._contact_cost_coeff = 0.5 * 1e-3 * 4 / CentipedeLegNum

        self.torso_geom_id = 1 + np.array(range(self.num_body)) * 5
        # make sure the centipede is not born to be end of episode
        self.body_qpos_id = 6 + 6 + np.array(range(self.num_body)) * 6
        self.body_qpos_id[-1] = 5
        self._get_joint_names()

        mujoco_env.MujocoEnv.__init__(self, self.xml_path, 5)

        utils.EzPickle.__init__(self)

    def _get_joint_names(self):
        infile = open(self.xml_path, 'r')
        xml_soup = bs(infile.read(), 'xml')

        # find the names of joints/bodies
        joints = xml_soup.find('worldbody').find_all('joint')
        self.joint_names = [joint['name'] for joint in joints]
        root_index = self.joint_names.index("root")
        del self.joint_names[root_index]

    def get_env_num_str(self, number):
        num_str = num2words.num2words(number)
        return num_str[0].upper() + num_str[1:]

    def step(self, a):
        self._action_val = a
        xposbefore = self.get_body_com("torso_" + str(self.num_body - 1))[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso_" + str(self.num_body - 1))[0]

        # calculate reward
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = self._control_cost_coeff * np.square(a).sum()
        contact_cost = self._contact_cost_coeff * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        # check if finished
        state = self.state_vector()
        notdone = np.isfinite(state).all() and \
                  self._check_height() and self._check_direction()
        done = not notdone
        # done = False

        obs = self._get_obs()

        return obs, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward
        )

    def _get_obs(self):
        """ Original code """
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def __get_obs(self):
        """
        My implementation of Obs including `Root`

        Description of Obs
            - body_xpos: Cartesian position of body frame     (nbody x 3)
            - body_xquat: Cartesian orientation of body frame (nbody x 4)
            - cvel: com-based velocity [3D rot; 3D tran]      (nbody x 6)
            - cinert: com-based body inertia and mass         (nbody x 10)

        :return
            - flat_obs = flattened observation
            - graph_obs = obs are organised by associating data with each node

        """
        flat_obs = np.concatenate([
            self.sim.data.body_xpos[1:, :].flat,
            self.sim.data.body_xquat[1:, :].flat,
            self.sim.data.cvel[1:, :].flat,
            self.sim.data.cinert[1:, :].flat
        ])
        # remove the "root"
        graph_obs = np.concatenate([
            self.sim.data.body_xpos[1:, :],
            self.sim.data.body_xquat[1:, :],
            self.sim.data.cvel[1:, :],
            self.sim.data.cinert[1:, :]
        ], axis=-1)
        return {"flat_obs": flat_obs,
                "graph_obs": graph_obs}

    def reset_model(self):
        while True:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-.1, high=.1
            )
            qpos[self.body_qpos_id] = self.np_random.uniform(
                size=len(self.body_qpos_id),
                low=-.1 / (self.num_body - 1),
                high=.1 / (self.num_body - 1)
            )

            qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
            self.set_state(qpos, qvel)
            if self._check_height() and self._check_direction():
                break
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.8
        body_name = 'torso_' + str(int(np.ceil(self.num_body / 2 - 1)))
        self.viewer.cam.trackbodyid = self.model.body_names.index(body_name)

    '''
    def _check_height(self):
        height = self.data.geom_xpos[self.torso_geom_id, 2]
        return (height < 1.5).all() and (height > 0.35).all()
    '''

    def _check_height(self):
        height = self.data.geom_xpos[self.torso_geom_id, 2]
        return (height < 1.15).all() and (height > 0.35).all()

    def _check_direction(self):
        y_pos_pre = self.data.geom_xpos[self.torso_geom_id[:-1], 1]
        y_pos_post = self.data.geom_xpos[self.torso_geom_id[1:], 1]
        y_diff = np.abs(y_pos_pre - y_pos_post)
        return (y_diff < 0.45).all()

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        """
        === Original Code ===
        i didn't change anything at all
        """

        if self.viewer:
            del self.viewer._markers[:]

        if mode == 'rgb_array':
            camera_id = None
            camera_name = 'track'
            if self.rgb_rendering_tracking and camera_name in self.model.camera_names:
                camera_id = self.model.camera_name2id(camera_name)
            """
            ======================
            This is where I change
            ======================
            """
            if self.viewer:
                self.add_joints_markers()
            self._get_viewer(mode).render(width, height, camera_id=camera_id)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            """
            ======================
            This is where I change
            ======================
            """
            if self.viewer:
                self.add_joints_markers()
            self._get_viewer(mode).render()

    def add_joints_markers(self):
        """ This adds the text annotation, which is associated with each joint, in the simulation
        To change the appearance of Text annotation, you can check the source code below.
        - https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjrendercontext.pyx

        - Note:
            Changeable attributes are shown L:232 to L:248 => (As of 31/07/2019)
        """
        for joint_name in self.joint_names:
            # for more APIs => https://github.com/openai/mujoco-py/blob/master/mujoco_py/generated/wrappers.pxi
            joint_id = self.sim.model.joint_name2id(joint_name)
            joint_pos = self.sim.data.body_xpos[joint_id, :]
            self.viewer.add_marker(pos=joint_pos,
                                   label="%s: %.3f" % (joint_name, self._action_val[joint_id - 1]),
                                   size=[0, 0, 0])


'''
    the following environments are just models with even number legs
'''


class CpCentipedeFourEnv(CentipedeEnv):

    def __init__(self):
        super(CentipedeFourEnv, self).__init__(CentipedeLegNum=4, is_crippled=True)


class CpCentipedeSixEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=6, is_crippled=True)


class CpCentipedeEightEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=8, is_crippled=True)


class CpCentipedeTenEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=10, is_crippled=True)


class CpCentipedeTwelveEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=12, is_crippled=True)


class CpCentipedeFourteenEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=14, is_crippled=True)


# regular


class CentipedeFourEnv(CentipedeEnv):

    def __init__(self):
        super(CentipedeFourEnv, self).__init__(CentipedeLegNum=4)


class CentipedeSixEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=6)


class CentipedeEightEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=8)


class CentipedeTenEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=10)


class CentipedeTwelveEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=12)


class CentipedeFourteenEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=14)


class CentipedeTwentyEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=20)


class CentipedeThirtyEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=30)


class CentipedeFortyEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=40)


class CentipedeFiftyEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=50)


class CentipedeOnehundredEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=100)


'''
    the following environments are models with odd number legs
'''


class CentipedeThreeEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=3)


class CentipedeFiveEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=5)


class CentipedeSevenEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=7)
