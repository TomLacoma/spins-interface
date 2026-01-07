import numpy as np
import random

from animation import *
from scipy.fft import fft, fftfreq
from scipy.spatial.transform import Rotation as R

gyro = 2.675221e8 #gyromagnetic ratio of a proton

def ret_none():
    return None

class Spins(Sim):
    """Handles a spins population in a constant magnetic field"""
    def __init__(self, x_coords, y_coords, z_coords, sim, phase = lambda x:0, peaks = None, temperature = 298, frequency = 300e6):
        """
        Handles a spin population and its updating over time using custom evolution functions
        Precomputes values.  

        Positional arguments : 
        :param list x_coords: list of x positions for the vectors anchors
        :param list y_coords: list of y positions for the vectors anchors
        :param list z_coords: list of z positions for the vectors anchors
        :param Sim sim: instance of Sim object to get all the timings and states from, see animation.py
        :param function phase: function returning the phase of a vector given its index i
        :param list peaks: list of the chemical shifts of the protons included in the simulation (in ppm) 
        :param dict pars: dictionary of optional parameters for the vectors precession frequency, equilibrium angle, magnitude
        """

        #Pass on the sim parameters
        self.sim = sim
        self.pars = sim.pars
        self._pars = sim._pars

        #Defines the atoms coordinates in 3d and their number
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.z_coords = z_coords
        self.n_x = len(x_coords) if not type(x_coords)==int else 1
        self.n_y = len(y_coords) if not type(y_coords)==int else 1
        self.n_z = len(z_coords) if not type(z_coords)==int else 1
        self.count = self.n_x*self.n_y*self.n_z

        #Computes the Boltzmann populations for the two states (alligned, 0 and antialligned, 1)
        #of a proton in a magnetic field
        self.weights = [0.5+np.exp((-6.63e-34*frequency)/1.38e-23*temperature), 0.5-np.exp((-6.63e-34*frequency)/1.38e-23*temperature)]
        #Randomly choose the protons spins states from the Boltzmann distribution
        self.states = random.choices([0,1], weights=self.weights, k=self.count)
        #The proton to be flipped, must be alligned (state 0)
        self.flippable = self.states.index(0)

        #Mesh of the x,y,z coordinates of protons
        self.x, self.y, self.z = np.meshgrid(x_coords, y_coords, z_coords)

        #Individual phase of each proton 
        self.phases = [phase(i) for i in range(self.count)]
        self.peaks = np.array([2, 4]) if not peaks else peaks
        #Defines protons precession frequencies based upon the peaks list content
        self.freq = [self.pars["f"]*(1+1e-6*self.peaks[i%len(self.peaks)]) for i in range(self.count)]

        
        self.updater = self.larmor #Default updater, basic Larmor precession
        self.prev_updater = self.updater #Memory of the last updater, useful for lifting sample

        self.spins = [np.array([0,0,1]) for i in range(self.count)] #Contains every spin vector coordinates


    def larmor(self, t, f, phi, state):
        """
        Updater for Larmor precession of magnetization.

        :param float t: next intended simulation time  
        :param float phi: invidual phase of spin
        :param int state : orientation of the spin in magnetic field (alligned=0, antiallignes=1)
        """
        u = self.pars["mag"]*np.sin(2*np.pi*f*t + phi)*np.sin(np.pi*self.pars["angle"]/180)
        v = self.pars["mag"]*np.cos(2*np.pi*f*t + phi)*np.sin(np.pi*self.pars["angle"]/180)
        w = (2*state-1)*self.pars["mag"]*np.cos(np.pi*self.pars["angle"]/180)
        return(np.array([u, v, w]))


    def lifted_updater (self, t, f, phi, state):
        """
        Updater for spins outside of magnetic field (random).

        :param float t: next intended simulation time  
        :param float phi: invidual phase of spin
        :param int state : orientation of the spin in magnetic field (alligned=0, antiallignes=1)
        """
        u = self.pars["mag"]*random.uniform(-1,1)
        v = self.pars["mag"]*random.uniform(-1,1)
        w = self.pars["mag"]*random.uniform(-1,1)
        try:
            return(np.array([u, v, w])/np.linalg.norm(np.array([u,v,w])))
        except:
            return([1,0,0])

    def update(self, t):
        """Computes the spins population at a given time t following the current updater"""
        for i in range(self.count):
            self.spins[i] = self.updater(t, self.freq[i], self.phases[i], self.states[i])



    def to_quiver(self):
        """Returns 6 lists containing the X, Y, Z, U, V, W components of each spin to pass to quiver functions"""
        u = np.array([i[0] for i in self.spins]).reshape(np.shape(self.x))
        v = np.array([i[1] for i in self.spins]).reshape(np.shape(self.y))
        w = np.array([i[2] for i in self.spins]).reshape(np.shape(self.z))
        return(self.x, self.y, self.z, u, v, w)

    def pulse(self):
        """Mimicks an NMR pulse, flips the flippable spins"""
        if self.sim.pulsed:
            self.sim.pulsed = False
            self.states[self.flippable] = 0
        else: 
            self.sim.pulsed = True
            self.states[self.flippable] = 1
    
    def lift(self):
        """Gets the sample out of the magnet, leading to no signal"""
        if self.sim.lifted:
            self.updater = self.prev_updater
            self.sim.lifted = False
        else:
            self.prev_updater = self.updater
            self.updater = self.lifted_updater
            self.sim.lifted = True

class Magnetization(Sim):
    """Handles bulk magnetization in a constant field B0"""
    def __init__(self, sim, M_default = np.array([0,0,2]), pulseprog = None):
        
        #Passed on simulation parameters
        self.sim = sim
        self.pars = sim.pars
        self._pars = sim._pars

        #Rotation object to handle constant frequency larmor precession
        self.z_precess = R.from_euler('z', self.sim._pars["timestep"]*2*np.pi*sim.pars["f"], degrees=False)

        #Multiplicator scalar for the transverse relaxation
        self.xy_decay = np.exp(-self.sim._pars["timestep"]/self.pars["T2"])

        #Transformation to be applied to the magnetization vector upon pulse, defaults to pi/2 hard pulse
        self.pulseprog = R.from_euler("x", 90, degrees = True) if not pulseprog else pulseprog

        #Asymptotic value for magnetization after relaxation
        self.M_inf = M_default
        #Starting values for magnetization
        self.M_xy = [0,0,0]
        self.M_z = self.M_inf
        self.M_z0 = self.M_z

        #Magnetization vector, reflects the magnetization at all time
        self.magnet = self.M_xy + self.M_z

        self.fid = np.dot(self.magnet, [1,0,0]) #x-projection of magnetization vector

    def update(self, t):
        """
        Updater function computing the next intended valued for magnetization, individual spins, etc. 
        Called at each frame prior to drawing.     
        """
        if self.sim.lifted:
            self.magnet = [0,0,0] #No bulk magnetization out of the magnet
            return
        if not self.sim.pulsed:
            self.magnet = self.M_inf #Shortcut to saying the spins are aligned woth B_0 when not pulsed
        else:
            #If you're here, congratulation, you've applied an NMR pulse to your sample ! 
            if self.sim.overflow:
                self.flip() #Check if the acquisition time is over and pulses again if so
            self.M_xy = self.z_precess.apply(self.M_xy)*self.xy_decay #Apply precession rotation and decay to transverse component
            self.M_z =  self.M_inf*(1-np.exp(-t/self.pars["T1"])) + self.M_z0*np.exp(-t/self.pars["T1"]) #Applies longitudinal decay 
            self.magnet = self.M_xy + self.M_z #Adds up the XY and Z components
        self.fid = np.dot(self.magnet, np.array([1,0,0])) #x-projection of magnetization vector

    
    def magnetization(self):
        """Magnetization vector as plottable by quiver object"""
        return(0, 0, 0, *self.magnet) 

    def pulse(self):
        """Sets all adequate variables to mimick an NMR pulse, see flip function for the math"""
        if self.sim.pulsed:
            self.sim.pulsed = False
            self.states[self.flippable] = 0
        else: 
            self.sim.pulsed = True
            self.flip()
            self.sim._pars["nscans"] = 0
    
    def flip(self):
        """
        Applies the transformation described by pulseprog to the magnetization 
        and updated the separated components (longitudinal, transverse)
        """
        self.magnet = self.pulseprog.apply(self.magnet)
        self.M_z0 = np.dot(self.magnet, [0,0,1])*np.array([0,0,1])
        self.M_xy = self.magnet - self.M_z0
        self.M_z = self.M_inf - self.M_z0

    def lift(self):
        """Lift the sample out of the spectrometer"""
        if self.sim.lifted:
            self.updater = self.prev_updater
            self.sim.lifted = False
        else:
            self.prev_updater = self.updater
            self.updater = self.sim.lifted_updater
            self.sim.lifted = True

class DispState:
    """
    Handles the different display states of the animated plot to display either 
    individual spins, the bulk magnetization, the fid or the Fourier transform
    """
    def __init__(self, state = "spins"):
        self.state_cnt = 0
        self._change = False
        
        self.state = state
        if not state in self.state:
            raise(ValueError)
        

        self.states_3d = ["spins", "mag"]
        self.states_2d = ["fid", "ft"]
        self.st_dict = {"spins":"Spins dans champs magnétique", "mag":"Aimantation totale", "fid":"Signal enregistré", "ft":"Spectre"}
        

    def is_2d(self):
        return self.state in self.states_2d
    
    def is_3d(self):
        return self.state in self.states_3d
    
    def mv(self, step):
        st = self.states_3d + self.states_2d
        self.state_cnt += step
        if self.state_cnt >= len(st):
            self.state_cnt = 0
        if self.state_cnt<0:
            self.state_cnt = len(st) - 1    
        self.state = st[self.state_cnt]
        self._change = True
    
    def title(self):
        return self.st_dict[self.state]