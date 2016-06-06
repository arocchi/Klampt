from controller import *
from klampt import *
import time
from soft_hand_loader import SoftHandLoader
import numpy as np


class SoftHandController(BaseController):
    """A controller for the SoftHand"""
    def __init__(self, robot, sim=None):
        if sim is not None:
            self.world = sim.world
            self.sim = sim
            self.sim.enableContactFeedbackAll()
        self.robot = robot
        self.startTime = None
        self.realStartTime = time.time()
        self.paramsLoader = SoftHandLoader('/home/user/Klampt/data/robots/soft_hand.urdf')

        self.hand = dict()
        self.mimic = dict()
        self.n_dofs = 0

        self.q_a_ref = 0.0

        self.K_p = 3.0
        self.K_d = 0.03
        self.K_i = 0.01
        self.q_a_int = 0.0

        self.K_p_m = 3.0
        self.K_d_m = 0.03

        self.synergy_reduction = 7.0  # convert cable tension into motor torque

        print "Loaded robot name is:", robot.getName()
        print "Number of Drivers:", robot.numDrivers()
        if robot.getName() == "soft_hand":
            self.n_dofs = robot.numDrivers()
            self.a_dofs = 1
        else:
            raise Exception('loaded robot is not a soft hand')

        self.u_to_l = []    # will contain a map from underactuated joint id (exluding mimics) to child link id
        self.l_to_i = dict()# will contain a map from link id (global) to link index (local)
        self.u_to_n = []    # will contain a map from underactuated joint id (excluding mimics) to driver id
        self.a_to_n = []    # will contain a map from actuated id to driver id
        self.m_to_n = []    # will contain a map from mimic joints id to driver id

        self.n_to_u = np.array(self.n_dofs*[-1])
        self.n_to_m = np.array(self.n_dofs*[-1])
        self.q_to_t = np.array(range(4,self.n_dofs+4))

        # loading previously defined maps
        for i in xrange(robot.numDrivers()):
            driver = robot.driver(i)
            print "Driver ", i, ": ", driver.getName()
            _,_,finger, phalanx,fake_id = driver.getName().split('_')
            if phalanx == "fake":
                if not self.mimic.has_key(finger):
                    self.mimic[finger] = []
                self.mimic[finger].append(i)
                self.m_to_n.append(i)
                m_id = len(self.m_to_n)-1
                self.n_to_m[i] = m_id
            elif phalanx == "wire":
                self.a_to_n.append(i)
            else:
                if not self.hand.has_key(finger):
                    self.hand[finger] = dict()
                self.hand[finger][phalanx] = i
                self.u_to_n.append(i)
                link = robot.link(robot.driver(i).getName())
                self.u_to_l.append(link.getID())
                self.l_to_i[link.getID()] = link.getIndex()
                u_id = len(self.u_to_n)-1
                self.n_to_u[i] = u_id

        self.u_dofs = len(self.u_to_n)
        self.m_dofs = len(self.m_to_n)

        # will contain a map from underactuated joint to mimic joints
        # this means, for example, that joint id 1 has to be matched by mimic joint 19
        self.m_to_u = self.m_dofs*[-1]

        for finger in self.hand.keys():
            for phalanx in self.hand[finger].keys():
                joint_count = 0
                if phalanx == 'abd':
                    continue
                else:
                    m_id = self.n_to_m[self.mimic[finger][joint_count]]
                    self.m_to_u[m_id] = self.n_to_u[self.hand[finger][phalanx]]
                    joint_count = joint_count+1

        # loading elasticity and reduction map
        self.R = np.array(self.u_dofs*[0.0]).T
        self.E = np.eye(self.u_dofs)

        for i in xrange(robot.numDrivers()):
            driver = robot.driver(i)
            _,_,finger, phalanx,fake_id = driver.getName().split('_')
            u_id = self.n_to_u[i]
            if u_id != -1:
                joint_position = self.paramsLoader.phalanxToJoint(finger,phalanx)
                self.R[u_id] = self.paramsLoader.handParameters[finger][joint_position]['r']
                self.E[u_id,u_id] = self.paramsLoader.handParameters[finger][joint_position]['e']

        print 'Soft Hand loaded.'
        print 'Mimic Joint Indices:', self.mimic
        print 'Underactuated Joint Indices:', self.hand
        print 'Joint parameters:', self.paramsLoader.handParameters
        print 'R:', self.R
        #self.E = 20 * self.E
        print 'E:', self.E

    def output(self, **inputs):
        api = ControllerAPI(inputs)
        t = api.time()
        if self.startTime is None:
            self.startTime = t
        t = t - self.startTime

        torque = np.array(self.n_dofs * [0.0])

        q = np.array(api.sensedConfiguration())

        q = q[self.q_to_t]
        dq = np.array(api.sensedVelocity())
        dq = dq[self.q_to_t]

        dq_a = dq[self.a_to_n]
        dq_u = dq[self.u_to_n]
        dq_m = dq[self.m_to_n]

        q_a = q[self.a_to_n]
        q_u = q[self.u_to_n]
        q_m = q[self.m_to_n]

        # command to close
        while self.q_a_ref < 1.0:
            self.q_a_ref = self.q_a_ref + 0.001

        R_E_inv_R_T_inv = 1.0 / (self.R.dot(np.linalg.inv(self.E)).dot(self.R.T))
        sigma = q_a # q_a goes from 0.0 to 1.0
        f_c, J_c = self.get_contact_forces_and_jacobians()
        tau_c = J_c.T.dot(f_c)

        f_a = R_E_inv_R_T_inv * sigma * self.synergy_reduction + R_E_inv_R_T_inv * np.linalg.inv(self.E).dot(tau_c)
        torque_a = self.K_p*(self.q_a_ref - q_a) \
                   + self.K_d*(0.0 - dq_a) \
                   + self.K_i*self.q_a_int \
                   - (f_a / self.synergy_reduction)

        torque_u = self.R.T*f_a - self.E.dot(q_u)
        torque_m = self.K_p_m*(q_u[self.m_to_u] - q_m) - self.K_d_m*dq_m

        torque[self.a_to_n] = torque_a
        torque[self.u_to_n] = torque_u
        torque[self.m_to_n] = torque_m

        #print 'q_u:', q_u
        #print 'q_a_ref-q_a:',self.q_a_ref-q_a
        #print 'q_u-q_m:', q_u[self.m_to_u]-q_m
        #print 'tau_u:', torque_u

        return api.makeTorqueCommand(torque)

    def get_contact_forces_and_jacobians(self):
        """ 
        Returns a force contact vector 1x(6*n_contacts)
        and a contact jacobian matrix 6*n_contactsxn.
        Contact forces are considered to be applied at the link origin
        """
        n_contacts = 0 # one contact per link
        if hasattr(self, 'world'):
            maxid = self.world.numIDs()
            J_l = dict()
            f_l = dict()
            t_l = dict()
            for l_id in self.u_to_l:
                l_index = self.l_to_i[l_id]
                link_in_contact = self.robot.link(l_index)
                contacts_per_link = 0
                for j in xrange(maxid): # TODO compute just one contact per link
                    contacts_l_id_j = len(self.sim.getContacts(l_id, j))                    
                    contacts_per_link += contacts_l_id_j
                    if contacts_l_id_j > 0:
                        if not f_l.has_key(l_id):
                            f_l[l_id] = self.sim.contactForce(l_id, j)
                            t_l[l_id] = self.sim.contactTorque(l_id, j)
                        f_l[l_id] = vectorops.add(f_l[l_id], self.sim.contactForce(l_id, j))
                        t_l[l_id] = vectorops.add(t_l[l_id], self.sim.contactTorque(l_id, j))
                        print "link", link_in_contact.getName(), """
                              in contact with obj""", self.world.getName(j), """
                              (""", len(self.sim.getContacts(l_id, j)), """
                              contacts)\n f=""",self.sim.contactForce(l_id, j), """
                              t=""", self.sim.contactTorque(l_id, j)

                ### debugging ###
                """
                if contacts_per_link == 0:
                    print "link", link_in_contact.getName(), "not in contact"
                """
                if contacts_per_link > 0:
                    n_contacts += 1
                    J_l[l_id] = np.array(link_in_contact.getJacobian(
                                        (0, 0, 0)))
                    print J_l[l_id].shape
        f_c = np.array(6 * n_contacts * [0.0])
        J_c = np.zeros((6 * n_contacts, self.u_dofs))

        for l_in_contact in xrange(len(J_l.keys())):
            f_c[l_in_contact * 6:l_in_contact * 6 + 3
                ] = f_l.values()[l_in_contact]
            f_c[l_in_contact * 6 + 3:l_in_contact * 6 + 6
                ] = t_l.values()[l_in_contact]
            J_c[l_in_contact * 6:l_in_contact * 6 + 6,
                :] = np.array(
                    J_l.values()[l_in_contact])[:, self.u_to_n]
        return (f_c, J_c)

def make(robot, sim=None):
    return SoftHandController(robot, sim)
