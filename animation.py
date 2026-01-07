import tkinter as tk

import numpy as np

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation


class Sim:
    """
    Basic class managing animation and several states flags and interactive plots, with switch timers for seamless transitions between displayed states
    """
    def __init__(self, pars = None, frame_pars = None):
        """
        :param dict pars : dictionary containing simulation parameters, expected : {"angle":float, "mag":float, "T1":float, "T2":float, "f":float, "atom_size":float}
        :param dict frame_pars: dictionary containing {"start_t":float, "end_t":float, "timestep":float}
        """
        self._pars = {"start_t":0, "end_t":3, "timestep":1/30, "TD0":1} if not frame_pars else frame_pars #Simulation parameters
        self.pars = {"angle":20, "mag":1, "T1":1, "T2":1, "f":4, "atom_size":0.1} if not pars else pars  #Physical parameters
        self.curr_t = 0 #Records the current simulation time
        self.switch_t = 0 #Records the last function switching time 
        self.overflow = False #Flags at True if the frames generator excedded the end_t value
        self.forced_overflow = False #Same but for external interrupt
        self.overflow_func = [self.nscans] #Functions to be executed at each overflow

        self.paused = False #Turns at True if the simulation is paused
        self.pulsed = False #Turns True if sample has been submitted to a pulse
        self.lifted = False #Turns true if the tube is removed from the magnet

        self.flags = [] #General purpose fkags, will be cleaned at the beggining of the next frame

    def __len__(self) -> int:
        """Returns the length of the time array of length end_t - start_t with timestep steps"""
        return int((self._pars["end_t"]-self._pars["start_t"])//self._pars["timestep"])
    
    def update_switch_t(self):
        """Records the time when updater function was switched"""
        self.switch_t = self.curr_t

    def frames(self):
        """Generator yielding every next intended time and updating the current simulation time"""
        _t = self._pars["start_t"]
        _step = self._pars["timestep"]
        self.overflow = False
        self.flags = []
        while True:
            self.overflow = False
            self.flags = []
            _t += _step
            if _t>self._pars["end_t"] or self.forced_overflow:
                _t = self._pars["start_t"]
                self.overflow = True
                self.switch_t = _t
                self.forced_overflow = False
                if len(self.overflow_func)>0:
                    for func in self.overflow_func:
                        func()
            self.curr_t = _t
            yield _t
    
    def flag(self, flag):
        """Sets a flag"""
        self.flags.append(flag)
    
    def unflag(self, flag):
        """Removes a flag"""
        if flag in self.flags:
            self.flags.remove(flag)

    def force_overflow(self):
        """Forces an overflow event"""
        self.forced_overflow = True

    def nscans(self):
        """Manages the number of scan loops before triggerring loop overflow and plotting the fft"""
        if "nscans" not in self._pars:
            self._pars["nscans"] = 0
        if "TD0" not in self._pars:
            self._pars["TD0"] = 1
        if self._pars["nscans"] >= self._pars["TD0"]:
            self._pars["nscans"] = 0
            self.flag("loop_overflow")
        else:
            self._pars["nscans"] += 1