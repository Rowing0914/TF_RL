## What is qpos/qvel?

- Someone said that 'q' in qpos and qvel refer to the generalize coordinate. The order of q is also dependent on the joint id you specify in the .xml model.
  - http://www.mujoco.org/forum/index.php?threads/qpos-qvel-meaning-of-freejoint.3658/

### With the running example

```python
 def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
```

- `self.sim.data.qpos` are the positions, with the first 7 element the 3D position (x,y,z) and orientation (quaternion x,y,z,w) of the torso and other joint parts
- `self.sim.data.qvel` are the velocities, with the first 6 elements the 3D velocity (x,y,z) and 3D angular velocity (x,y,z) and other joint parts
- `cfrc_ext` are the external forces (force x,y,z and torque x,y,z) applied to each of the links at the center of mass. For the Ant, this is 14*6: the ground link, the torso link, and 12 links for all legs (3 links for each leg)
  - https://github.com/openai/gym/issues/585

### [What does MuJoCo Official say?](<http://www.mujoco.org/book/>)

#### Joint coordinates

One of the key distinctions between MuJoCo and gaming engines (such as ODE, Bullet, Havoc, PhysX) is that MuJoCo operates in generalized or joint coordinates, while gaming engines operate in Cartesian coordinates, although Bullet now supports generalized coordinates. The differences between these two approaches can be summarized as follows: 

Joint coordinates:

- Best suited for elaborate kinematic structures such as robots;
- Joints add degrees of freedom among bodies that would be welded together by default;
- Joint constraints are implicit in the representation and cannot be violated;
- The positions and orientations of the simulated bodies are obtained from the generalized coordinates via forward kinematics, and cannot be manipulated directly (except for root bodies).

Cartesian coordinates:

- Best suited for many bodies that bounce off each other, as in molecular dynamics and box stacking;
- Joints remove degrees of freedom among bodies that would be free-floating by default;
- Joint constraints are enforced numerically and can be violated;
- The positions and orientations of the simulated bodies are represented explicitly and can be manipulated directly, although this can introduce further joint constraint violations.

Joint coordinates can be particularly confusing when working with free-floating bodies that are part of a model which also contains kinematic trees. This is clarified below.

### Source code of mjdata.pxd

<https://github.com/openai/mujoco-py/blob/master/mujoco_py/pxd/mjdata.pxd>

## [Source code of MuModel](<https://github.com/openai/mujoco-py/blob/c5f60322467ec8ecc0db64c3e18a4da762c27e45/mujoco_py/pxd/mjmodel.pxd>)

```python
int nq                          # number of generalized coordinates = dim(qpos)
int nv                          # number of degrees of freedom = dim(qvel)
int nu                          # number of actuators/controls = dim(ctrl)
int na                          # number of activation states = dim(act)
int nbody                       # number of bodies
int njnt                        # number of joints
```

