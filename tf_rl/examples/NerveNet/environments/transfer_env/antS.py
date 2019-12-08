# -----------------------------------------------------------------------------
#   @brief:
#       The slim ant environments.
#   @author:
#       Tingwu (Wilson) Wang, Aug. 30nd, 2017
# -----------------------------------------------------------------------------

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import environments.init_path as init_path
import os
from bs4 import BeautifulSoup as bs

DEFAULT_SIZE = 500

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        xml_path = os.path.join(init_path.get_base_dir(),
                                'environments', 'assets',
                                'ant.xml')
        self.xml_path = xml_path
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        """ Same as https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py#L10 """
        self._action_val = a
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        """ original code """
        return np.concatenate([
            self.sim.data.body_xpos[2:, :].flat,
            self.sim.data.body_xquat[2:, :].flat,
            self.sim.data.cvel[2:, :].flat,
            self.sim.data.cinert[2:, :].flat
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

        """
        10 is for the bodies below
        'world', 'torso', 'aux_1', 'f_1', 'aux_2', 'f_2', 'aux_3', 'f_3', 'aux_4', 'f_4'
        So that we remove first 2 items since it doesn't relate to the actuator
        """
        flat_obs = np.concatenate([
            self.sim.data.body_xpos[2:, :].flat,
            self.sim.data.body_xquat[2:, :].flat,
            self.sim.data.cvel[2:, :].flat,
            self.sim.data.cinert[2:, :].flat
        ])
        # remove the "root"
        graph_obs = np.concatenate([
            self.sim.data.body_xpos[2:, :],
            self.sim.data.body_xquat[2:, :],
            self.sim.data.cvel[2:, :],
            self.sim.data.cinert[2:, :]
        ], axis=-1)
        return {"flat_obs": flat_obs,
                "graph_obs": graph_obs}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

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
        for actuator_name in self.sim.model.actuator_names:
            # for more APIs => https://github.com/openai/mujoco-py/blob/master/mujoco_py/generated/wrappers.pxi
            actuator_id = self.sim.model.joint_name2id(actuator_name)
            actuator_pos = self.sim.data.get_joint_xanchor(actuator_name)
            self.viewer.add_marker(pos=actuator_pos,
                                   label="%s: %.3f" % (actuator_name, self._action_val[actuator_id - 1]),
                                   size=[0, 0, 0])


class AntWithGoalEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        xml_path = os.path.join(init_path.get_base_dir(),
                                'environments', 'assets',
                                'ant_with_goal.xml')
        self.radius = 2.0
        self.xml_path = xml_path
        self._set_goal()
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)
        self._goal_idx = self.sim.model.site_name2id("goal")  # internal usage to randomly reset the location

        """
        For reproducibility, you might want to open these lines to set the goals in advance
        
        self.train_goals = pickle.load(open('./environments/assets/goals/ant_train.pkl', "rb"))
        self.eval_goals = pickle.load(open('./environments/assets/goals/ant_eval.pkl', "rb"))        
        """

        self._get_joint_names()

    def _get_joint_names(self):
        infile = open(self.xml_path, 'r')
        xml_soup = bs(infile.read(), 'xml')

        # find the names of joints/bodies
        joints = xml_soup.find('worldbody').find_all('joint')
        self.joint_names = [joint['name'] for joint in joints]
        root_index = self.joint_names.index("root")
        del self.joint_names[root_index]

    def _set_goal(self):
        self.goal = self._generate_goal()

    def _generate_goal(self):
        """ Randomly generate the goal within the pre-defined radius """
        angle = np.random.uniform(0, np.pi)
        xpos = self.radius * np.cos(angle)
        ypos = self.radius * np.sin(angle)
        return np.array([xpos, ypos], dtype=np.float32)

    def step(self, a):
        self._action_val = a
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        xyposafter = self.get_body_com("torso")[:2]

        if np.linalg.norm(xyposafter - self.goal) > 0.8:
            goal_reward = -np.sum(np.abs(self.goal)) + 4.0
        else:
            goal_reward = -np.sum(np.abs(xyposafter - self.goal)) + 4.0  # make it happy, not suicidal

        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        # reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        reward = goal_reward - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal=self.goal,
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        """ original code """
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

        """
        10 is for the bodies below
        'world', 'torso', 'aux_1', 'f_1', 'aux_2', 'f_2', 'aux_3', 'f_3', 'aux_4', 'f_4'
        So that we remove first 2 items since it doesn't relate to the actuator
        """

        flat_obs = np.concatenate([
            self.sim.data.body_xpos[2:, :].flat,
            self.sim.data.body_xquat[2:, :].flat,
            self.sim.data.cvel[2:, :].flat,
            self.sim.data.cinert[2:, :].flat
        ])
        # remove the "root"
        graph_obs = np.concatenate([
            self.sim.data.body_xpos[2:, :],
            self.sim.data.body_xquat[2:, :],
            self.sim.data.cvel[2:, :],
            self.sim.data.cinert[2:, :]
        ], axis=-1)
        return {"flat_obs": flat_obs,
                "graph_obs": graph_obs}

    def reset_model(self):
        self._set_goal()  # when we reset the model, we reset the goal as well.
        self.sim.model.site_pos[self._goal_idx][0] = self.goal[0]
        self.sim.model.site_pos[self._goal_idx][1] = self.goal[1]
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

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