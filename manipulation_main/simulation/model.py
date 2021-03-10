import pybullet as p


class Model(object):
    def __init__(self, physics_client):
        self._physics_client = physics_client
        self.joints = []
        self.links = []
        self.jointPositions = []

    def load_model(self, path, start_pos=[0, 0, 0],
                   start_orn=[0, 0, 0, 1], scaling=1., static=False):
        if path.endswith('.sdf'):
            self.model_id = self._physics_client.loadSDF(path, globalScaling=scaling)[0]
            self._physics_client.resetBasePositionAndOrientation(self.model_id, start_pos, start_orn)
            self.jointPositions = [0.00289251589762636,
                                   0.7956459404310455,
                                   -0.0071792792711452185,
                                   -0.7824793442451049,
                                   0.005132245553240464,
                                   1.5634707329864685,
                                   -0.002680634976774496,
                                   6.176210963187958e-06,
                                   -0.05499978684942884,
                                   -4.854655260695386e-06,
                                   -4.812201245188288e-05,
                                   1.8108007322371046e-05,
                                   -9.701091068665463e-07,
                                   0,
                                   0]
            self.num_joints = self._physics_client.getNumJoints(self.model_id)
            for joint_index in range(self.num_joints):
                self._physics_client.resetJointState(self.model_id, joint_index,
                                                     self.jointPositions[joint_index])
                self._physics_client.setJointMotorControl2(self.model_id,
                                                           joint_index,
                                                           self._physics_client.POSITION_CONTROL,
                                                           targetPosition=self.jointPositions[joint_index],
                                                           force=100)
        else:
            self.model_id = self._physics_client.loadURDF(
                path, start_pos, start_orn,
                globalScaling=scaling, useFixedBase=static)
            self.num_joints = self._physics_client.getNumJoints(self.model_id)

        joints, links = {}, {}
        for i in range(self.num_joints):
            joint_info = self._physics_client.getJointInfo(self.model_id, i)
            joint_limits = {'lower': joint_info[8], 'upper': joint_info[9],
                            'force': joint_info[10]}
            joints[i] = _Joint(self._physics_client, self.model_id, i, joint_limits)
            links[i] = _Link(self._physics_client, self.model_id, i)
        self.joints, self.links = joints, links

        return self.model_id

    def get_joints(self):
        for i in range(self._physics_client.getNumJoints(self.model_id)):
            self.joints[i] = self._physics_client.getJointInfo(self.model_id, i)

    def get_pose(self):
        """Return the pose of the model base."""
        _, _, _, _, world_pos, world_orn = self._physics_client.getLinkState(self.model_id, 6)

        return world_pos, world_orn

    def get_pose_cam(self):
        """Return the pose of the model base."""
        pos, orn, _, _, _, _ = self._physics_client.getLinkState(self.model_id, 7)

        return pos, orn

    def get_base(self):
        return self._physics_client.getBasePositionAndOrientation(self.model_id)


class _Link(object):
    def __init__(self, physics_client, model_id, link_id):
        self._physics_client = physics_client
        self.model_id = model_id
        self.lid = link_id

    def get_pose(self):
        link_state = p.getLinkState(self.model_id, self.lid)
        position, orientation = link_state[0], link_state[1]
        return position, orientation


class _Joint(object):
    def __init__(self, physics_client, model_id, joint_id, limits):
        self._physics_client = physics_client
        self.model_id = model_id
        self.jid = joint_id
        self.limits = limits

    def get_position(self):
        joint_state = self._physics_client.getJointState(
            self.model_id, self.jid)
        return joint_state[0]

    def set_position(self, position, max_force=100.):
        self._physics_client.setJointMotorControl2(
            self.model_id, self.jid,
            controlMode=p.POSITION_CONTROL,
            targetPosition=position,
            force=max_force)

    def disable_motor(self):
        self._physics_client.setJointMotorControl2(
            self.model_id, self.jid, controlMode=p.VELOCITY_CONTROL, force=0.)
