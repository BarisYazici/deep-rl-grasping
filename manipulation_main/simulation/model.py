import pybullet as p
import numpy as np


class Model(object):
    def __init__(self, physics_client):
        self._physics_client = physics_client

    def load_model(self, path, start_pos=[0, 0, 0], 
                   start_orn=[0, 0, 0, 1], scaling=1., static=False):
        if path.endswith('.sdf'):
            model_id = self._physics_client.loadSDF(path, globalScaling=scaling)[0]
            self._physics_client.resetBasePositionAndOrientation(model_id, start_pos, start_orn)
        else:
            model_id = self._physics_client.loadURDF(
                path, start_pos, start_orn,
                globalScaling=scaling, useFixedBase=static)        
        self.model_id = model_id
        # self._get_limits(self.model_id)
        joints, links = {}, {}
        for i in range(self._physics_client.getNumJoints(self.model_id)):
            joint_info = self._physics_client.getJointInfo(self.model_id, i)
            # joint_name = joint_info[1].decode('utf8')
            joint_limits = {'lower': joint_info[8], 'upper': joint_info[9],
                            'force': joint_info[10]}
            joints[i] = _Joint(self._physics_client, self.model_id, i, joint_limits)
            # link_name = joint_info[12].decode('utf8')
            links[i] = _Link(self._physics_client, self.model_id, i)
        self.joints, self.links = joints, links

        return model_id

    def get_joints(self):
        for i in range(self._physics_client.getNumJoints(self.model_id)):
            self.joints[i] = self._physics_client.getJointInfo(self.model_id, i)

    def get_pose(self):
        """Return the pose of the model base."""
        pos, orn, _, _, _, _ = self._physics_client.getLinkState(self.model_id, 3)
        return (pos, orn)
    
    def getBase(self):
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