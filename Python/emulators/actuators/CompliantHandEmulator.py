from klampt import vectorops
from klampt.simulation import ActuatorEmulator
import numpy as np

class CompliantHandEmulator(ActuatorEmulator):
    """An simulation model for the SoftHand for use with SimpleSimulation"""
    def __init__(self, sim, robotindex=0, link_offset=0, driver_offset=0, a_dofs=0, d_dofs=0, u_dofs=0, m_dofs=0):
        self.world = sim.world
        self.sim = sim
        self.sim.enableContactFeedbackAll()
        self.controller = self.sim.controller(robotindex)
        self.robot = self.world.robot(robotindex)

        self.link_offset = link_offset
        self.driver_offset = driver_offset

        self.hand = dict()
        self.mimic = dict()
        self.n_dofs = 0

        self.q_a_int = 0.0

        self.synergy_reduction = 1.0  # convert cable tension into motor torque
        self.effort_scaling = -1.0

        self.n_dofs = self.robot.numDrivers()
        self.a_dofs = a_dofs
        self.d_dofs = d_dofs
        self.u_dofs = u_dofs
        self.m_dofs = m_dofs

        self.u_to_l = []  # will contain a map from underactuated joint id (excluding mimics) to child link id
        self.l_to_i = dict()  # will contain a map from link id (global) to link index (local)
        self.u_to_n = []  # will contain a map from underactuated joint id (excluding mimics) to driver id
        self.a_to_n = []  # will contain a map from synergy actuators id to driver id
        self.d_to_n = []  # will contain a map from regular actuators id to driver id
        self.m_to_n = []  # will contain a map from mimic joints id to driver id

        self.n_to_u = np.array(self.n_dofs * [-1])
        self.n_to_m = np.array(self.n_dofs * [-1])
        self.n_to_a = np.array(self.n_dofs * [-1])
        self.n_to_d = np.array(self.n_dofs * [-1])

        self.q_to_t = []  # maps active drivers to joint ids
        # (basically removes weld joints, counts affine joints only once, takes into account
        #  floating base and regular joints properly)

        # used to apply virtual forces to fingers - used for debugging purposes
        self.virtual_contacts = dict()
        self.virtual_wrenches = dict()

        for i in xrange(self.robot.numDrivers()):
            driver = self.robot.driver(i)
            link = self.robot.link(driver.getName())
            self.q_to_t.append(link.getIndex())

        self.m_to_u = self.m_dofs * [-1]

        # loading elasticity and reduction map
        self.R = np.zeros((self.a_dofs, self.u_dofs))
        self.E = np.eye(self.u_dofs)

        self.q_a_ref = np.array(self.a_dofs * [0.0])
        self.q_d_ref = np.array(self.d_dofs * [0.0])

        self.loadHandParameters()

        self.loadContactInfo()

        self.setupController()

        self.printHandInfo()

    def loadHandParameters(self):
        """
        loadHandParameters loads the maps from:
         - underactuated joint id to driver id and vice_versa (n_to_u, u_to_n)
         - synergy actuators to driver id and vice_versa (a_to_n, n_to_a)
         - regular actuators to driver id and vice_versa (d_to_n, n_to_d)
         - mimic joints to driver id and vice-versa (m_to_n, n_to_m)

         Notice that we expect the hand model to respect the standard defined in the documentation, in particular:
         - underactuated joints should have as a child a real link with corresponding collision mesh.
         Still, if this characteristic is not satisfied, it is possible to manually write the map that links underactuated
         joint ids with the following collision mesh, that is the the u_to_l map. For each link in the map, the l_to_i
         map needs also to be filled.
        """
        pass

    def updateR(self, q_u):
        return self.R

    def loadContactInfo(self):
        # loading previously defined maps
        for i in self.u_to_n:
            link = self.robot.link(self.robot.driver(i).getName())
            self.u_to_l.append(link.getID())
            self.l_to_i[link.getID()] = link.getIndex()

    def setupController(self):
        kP, kI, kD = self.controller.getPIDGains()
        for i in self.u_to_n:
            kP[i] = 0.0
            kI[i] = 0.0
            kD[i] = 0.0
        self.controller.setPIDGains(kP, kI, kD)

    def printHandInfo(self):
        print 'Actuated Joint Indices:', self.d_to_n
        print 'Synergy Joint Indices:', self.a_to_n
        print 'Underactuated Joint Indices:', self.u_to_n
        print 'Mimic Joint Indices:', self.m_to_n
        print 'Joint to Driver map:', self.q_to_t
        print 'R:', self.R
        print 'E:', self.E

    def output(self):
        """
        @return (torque, qdes) where #torque = n_dofs, #qdes = n_links
        """
        torque = np.array(self.n_dofs * [0.0])
        g_q = np.array(self.robot.getGravityForces([0,0,-9.81]))
        # gravity compensation
        torque = g_q[self.q_to_t]

        q = np.array(self.controller.getSensedConfig())

        q = q[self.q_to_t]
        dq = np.array(self.controller.getSensedVelocity())
        dq = dq[self.q_to_t]

        dq_a = dq[self.a_to_n]
        dq_u = dq[self.u_to_n]
        dq_m = dq[self.m_to_n]

        q_a = q[self.a_to_n]
        q_u = q[self.u_to_n]
        q_m = q[self.m_to_n]

        # updates self.R
        self.updateR(q_u)

        E_inv = np.linalg.inv(self.E)
        R_E_inv_R_T_inv = np.linalg.inv(self.R.dot(E_inv).dot(self.R.T))
        sigma = q_a  # q_a goes from 0.0 to 1.0
        f_c, J_c = self.get_contact_forces_and_jacobians()
        tau_c = J_c.T.dot(f_c)

        # tendon tension
        f_a =  self.effort_scaling * R_E_inv_R_T_inv.dot(self.R).dot(E_inv).dot(tau_c) + self.synergy_reduction * R_E_inv_R_T_inv.dot(sigma)

        torque_a = - (f_a / self.synergy_reduction) # f_a offset

        torque_u = self.R.T.dot(f_a) - self.E.dot(q_u)

        torque_m = len(self.m_to_u)*[0.0] # 0 offset

        q_u_ref = self.effort_scaling * (-E_inv + E_inv.dot(self.R.T).dot(R_E_inv_R_T_inv).dot(self.R).dot(E_inv)).dot(tau_c) + E_inv.dot(self.R.T).dot(R_E_inv_R_T_inv).dot(sigma) * self.synergy_reduction

        torque[self.a_to_n] = torque_a # synergy actuators are affected by gravity
        torque[self.u_to_n] += torque_u # underactuated joints are emulated, no gravity
        torque[self.m_to_n] += torque_m # mimic joints are emulated, no gravity

        qdes = np.array(self.controller.getCommandedConfig())
        qdes[[self.q_to_t[u_id] for u_id in self.u_to_n]] = q_u_ref
        qdes[[self.q_to_t[m_id] for m_id in self.m_to_n]] = q_u_ref
        qdes[[self.q_to_t[a_id] for a_id in self.a_to_n]] = self.q_a_ref
        qdes[[self.q_to_t[d_id] for d_id in self.d_to_n]] = self.q_d_ref

        # print 'q_u:', q_u
        # print 'q_a_ref-q_a:',self.q_a_ref-q_a
        # print 'q_u-q_m:', q_u[self.m_to_u]-q_m
        # print 'tau_u:', torque_u

        # quirk: torque has n_dofs elements, qdes has n_links elements.
        # setPIDCommand accepts a qdes of either n_links or n_dofs size, but
        # requires a torque sized as n_dofs. We will therefore return the full
        # qdes so that we can use controller.getCommandedVelocity() as velocity term of the PID
        # (which returns a vector sized n_links)
        return torque, qdes

    def initR(self):
        q = np.array(self.controller.getSensedConfig())
        q = q[self.q_to_t]
        q_u = q[self.u_to_n]
        self.updateR(q_u)

    def get_contact_forces_and_jacobians(self):
        """
        Returns a force contact vector 1x(6*n_contacts)
        and a contact jacobian matrix 6*n_contactsxn.
        Contact forces are considered to be applied at the link origin
        """
        n_contacts = 0  # one contact per link
        maxid = self.world.numIDs()
        J_l = dict()
        f_l = dict()
        t_l = dict()
        for l_id in self.u_to_l:
            l_index = self.l_to_i[l_id]
            link_in_contact = self.robot.link(l_index)
            contacts_per_link = 0
            for j in xrange(maxid):  # TODO compute just one contact per link
                contacts_l_id_j = len(self.sim.getContacts(l_id, j))
                contacts_per_link += contacts_l_id_j
                if contacts_l_id_j > 0:
                    if not f_l.has_key(l_id):
                        print "+"
                        f_l[l_id] = self.sim.contactForce(l_id, j)
                        t_l[l_id] = self.sim.contactTorque(l_id, j)
                    else:
                        f_l[l_id] = vectorops.add(f_l[l_id], self.sim.contactForce(l_id, j))
                        t_l[l_id] = vectorops.add(t_l[l_id], self.sim.contactTorque(l_id, j))
                    ### debugging ###
                    # print "link", link_in_contact.getName(), """
                    #      in contact with obj""", self.world.getName(j), """
                    #      (""", len(self.sim.getContacts(l_id, j)), """
                    #      contacts)\n f=""",self.sim.contactForce(l_id, j), """
                    #      t=""", self.sim.contactTorque(l_id, j)
            if self.virtual_contacts.has_key(l_id):
                if not f_l.has_key(l_id):
                    f_l[l_id] = self.virtual_wrenches[l_id][0:3]
                    t_l[l_id] = self.virtual_wrenches[l_id][3:6]
                else:
                    f_l[l_id] = vectorops.add(f_l[l_id], self.virtual_wrenches[l_id][0:3])
                    t_l[l_id] = vectorops.add(t_l[l_id], self.virtual_wrenches[l_id][3:6])


            ### debugging ###
            """
            if contacts_per_link == 0:
                print "link", link_in_contact.getName(), "not in contact"
            """
            if contacts_per_link > 0 or self.virtual_contacts.has_key(l_id):
                n_contacts += 1
                J_l[l_id] = np.array(link_in_contact.getJacobian(
                    (0, 0, 0)))
                # print J_l[l_id].shape
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


    def setCommand(self, command):
        self.q_a_ref = [max(min(v, 1), 0) for i, v in enumerate(command) if i < self.a_dofs]
        self.q_d_ref = [max(min(v, 1), 0) for i, v in enumerate(command) if i >= self.a_dofs and i < self.a_dofs + self.d_dofs]


    def getCommand(self):
        return np.hstack([self.q_a_ref, self.q_d_ref])


    def process(self, commands, dt):
        if commands:
            if 'position' in commands:
                self.setCommand(commands['position'])
                del commands['position']
            if 'qcmd' in commands:
                self.setCommand(commands['qcmd'])
                del commands['qcmd']
            if 'speed' in commands:
                pass
            if 'force' in commands:
                pass

        torque, qdes = self.output()
        dqdes = self.controller.getCommandedVelocity()
        self.controller.setPIDCommand(qdes, dqdes, torque)

    def substep(self, dt):
        torque, qdes = self.output()
        #qdes = np.array(self.controller.getCommandedConfig())
        qdes[[self.q_to_t[d_id] for d_id in self.d_to_n]] = self.q_d_ref
        dqdes = self.controller.getCommandedVelocity()
        self.controller.setPIDCommand(qdes, dqdes, torque)


    def drawGL(self):
        pass