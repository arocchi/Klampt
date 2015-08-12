from controller import *
from klampt import *
import time
import numpy as np
from urdf_parser_py.urdf import URDF

import pYTask
import ExampleKlamptController as OpenSoT
from collections import Counter

class KlamptJointInfo(object):
    def __init__(self, robot_klampt, robot_urdf_path):
        self.robot = robot_klampt
        self.urdf = URDF.from_xml_file(robot_urdf_path)

    def jntMapToKlampt(self, jntMap):
        q = np.zeros(self.robot.numDrivers())
        for i in xrange(self.robot.numDrivers()):
            link = self.robot.getDriver(i).getName()
            parent_joint, parent_link = self.urdf.parent_map[link]
            q[i] = jntMap[parent_joint]
        return q

    def klamptToJntMap(self, joints):
        assert(joints is not None)
        assert(len(joints) == self.robot.numDrivers())
        jntMap = dict()
        for i in xrange(self.robot.numDrivers()):
            link = self.robot.getDriver(i).getName()
            parent_joint, parent_link = self.urdf.parent_map[link]
            from IPython.core.debugger import Tracer
            Tracer()()
            jntMap[parent_joint] = joints[i]
        return OpenSoT.KlamptController.JntMap(jntMap)

class HuboPlusController(BaseController):
    """A controller for the SoftHand"""
    def __init__(self,robot):
        self.robot = robot
        self.startTime = None
        self.realStartTime = time.time()
        self.controller = OpenSoT.ExampleKlamptController()
        self.posture = None
        # TODO change absolute path with relative, take into account argv[0]
        self.jntMapper = KlamptJointInfo(self.robot, '/home/motion/Klampt/data/robots/huboplus/huboplus.urdf')

    def output(self,**inputs):
        api = ControllerAPI(inputs)
        t = api.time()

        if self.startTime is None:
            self.startTime = t

        if self.posture is None:
            q = api.sensedConfiguration()
            if q is not None:
                self.posture = self.jntMapper.klamptToJntMap(q)
                self.controller.setPosture(self.posture)
            else:
                print "Error: sensedConfiguration is empty"

        t = t - self.startTime

        # Main loop - not CLIK
        if self.posture is not None:
            dq = self.controller.computeControl(self.posture)
            self.posture = OpenSoT.JntMap(Counter(self.posture) + Counter(dq))

            q_cmd = self.jntMapper.jntMapToKlampt(self.posture)

            print q_cmd
        else:
            q_cmd = np.zeros(self.robot.numDrivers())

        return api.makePositionCommand(q_cmd)

def make(robot):
    return HuboPlusController(robot)
