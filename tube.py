from spins import *
import tkinter as tk

from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

import sys

class Tube:
    """Tube object handling the physics in every display state, the UI and animated plots using tkinter"""
    def __init__(self, sim, spins, M, state):
        """
        :param Sim sim: Sim object handling the time dependant simulation
        :param Spins spins: Spins object describing the individual spins
        :param Magnetization M: Magnetization object describing the bulk magnetization
        :param DispState state: DispState object handling (you guessed it) the currently displayed state
        """
        self.sim = sim
        self.spins = spins
        self.M = M
        self.state = state

        self.show_axis = False

        self.t_arr, self.fid_arr = [spins._pars["start_t"]], [M.fid] #time and fid storing arrays
        self.freq, self.ft_vals = [0], [0] #freq and FT storage arrays
        self.mag_vals=[] #values taken by magnetization vector

        self.wm = lambda t:np.exp(-t*0.3) #exponential, 0.3Hz line broadening

        self.fig = Figure() #figure environment embedded in tk
        
        #Created a tkinter window
        self.root = tk.Tk()
        self.root.configure(bg='white')
        self.root.wm_title("SPINS v0.1")

        self.mainloop = self.root.mainloop
        #Creates a canva for embedding the matplotlib mess
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # A tk.DrawingArea.
        self.canvas.draw()

        #Creates the UI buttons...
        self.next_btn = tk.Button(master=self.root, text="Next", command=self.next_fig)
        self.switch_btn = tk.Button(master=self.root, text = "Acquire !", command = self.pulse)
        self.prev_btn = tk.Button(master=self.root, text="Previous", command=self.prev_fig)
        self.lift_btn = tk.Button(master=self.root, text="Lift sample", command=self.toggle_lift)
        #...and add them to the UI
        self.canvas.get_tk_widget().grid(column=2, row=1, sticky="wens")
        self.next_btn.grid(column = 4, row=2)
        self.switch_btn.grid(column = 3, row=2)
        self.lift_btn.grid(column = 2, row=2)
        self.prev_btn.grid(column = 1, row=2)
        #Animation function for the matplotlib animation, crucial
        self.ani = FuncAnimation(self.fig, self.update, frames = sim.frames, interval = spins._pars["timestep"], cache_frame_data=False)
        #Handles the windows clising event preventing softlocks
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
        #Different mpl artists for different purposes
        self.ax, self.artist, self.atoms, self.mag_trace = None, None, None, None
        self.create_ax()
        #fig.patch.set_facecolor("black")
        #ax.set_facecolor("black")

    def cleanup(self):
        """Clears the programs upon exit"""
        plt.close(self.fig)
        self.root.destroy()
        sys.exit(1)

    def next_fig(self):
        """Next display state, can overflow"""
        self.state.mv(1)
        if self.state.state == 'fid':
            self.ax.set_ylim(-2,2)

    def prev_fig(self):
        """Previous display state, can overflow"""
        self.state.mv(-1)
        if self.state.is_2d():
            self.ax.set_ylim(-2,2)

    def pulse(self):
        """Toggles a pulse"""
        self.spins.pulse()
        self.switch_btn.config(text="Idle" if self.sim.pulsed else "Acquire!")
        self.sim.force_overflow()
        self.mag_vals = []

    def toggle_lift(self):
        """Toggles lift"""
        self.spins.lift()
        self.lift_btn.config(text="Lower sample" if self.sim.lifted else "Lift sample")

    def lift(self):
        """Force lift upp"""
        if not self.sim.lifted:
            self.spins.lift()
            self.lift_btn.config(text="Lower sample" if self.sim.lifted else "Lift sample")
    
    def unlift(self):
        """Force lift down"""
        if self.sim.lifted:
            self.spins.lift()
            self.lift_btn.config(text="Lower sample" if self.sim.lifted else "Lift sample")

            
    def update(self, t):
        """
        The big messy updater function, responsible for computing the right data depending on display state, 
        updating it on screen, and handling the user inputs as well as overflows and timed events from Sim.
        """
        if self.sim.overflow: #In case of time overflow
            if "loop_overflow" in self.sim.flags and self.sim.pulsed:
            #Means all the acquisition loops has been done
                self.spins.pulse() #Switches to idle
                self.switch_btn.config(text="Idle" if self.sim.pulsed else "Acquire!")
                self.ft() #Computes the Fourier transform of the FID
            #Resets the saved time, fid and magnetization values
            self.t_arr = []
            self.fid_arr = []
            self.mag_vals = []

        if self.state._change: #On changing the display state
            self.fig.clf()
            self.create_ax() #Resets the figure with the right shape
            self.state._change = False
            if self.state.state == "fid":
                self.ax.set_xlim(self.sim._pars["start_t"], self.sim._pars["end_t"])
        
        if self.state.is_3d(): #In case the disp state changes to 3D
            try:
                self.artist.remove()
            except:
                pass
            if not self.show_axis:
                self.ax.set_axis_off()

        #Legends the graph    
        self.ax.set_title(self.state.title())
        #Updates all of the physics
        self.M.update(t)
        self.spins.update(t)
        self.t_arr.append(self.sim.curr_t)
        self.fid_arr.append(self.M.fid)
        self.mag_vals.append(self.M.magnet)

        #Big array of conditions plotting the rigth values depending of the disp state
        if self.state.state == "spins":
            self.artist = self.ax.quiver(*self.spins.to_quiver())
            if not self.atoms:
                self.atoms, = self.ax.plot(self.spins.x, self.spins.y, self.spins.z, linestyle="", marker="o", color = "gray")
        if self.state.state == "mag":
            self.artist = self.ax.quiver(*self.M.magnetization(), linewidth = 2)
            self.mag_trace.set_data_3d([m[0] for m in self.mag_vals], [m[1] for m in self.mag_vals], [m[2] for m in self.mag_vals])
        if self.state.state == "fid": 
            self.artist.set_data(self.t_arr, self.fid_arr)
            self.ax.set_xlim(self.sim._pars["start_t"], self.sim._pars["end_t"])
        if self.state.state == "ft":
            self.artist.set_data(self.freq, self.ft_vals)
            self.ax.set_xlim(1.1*min(self.freq), 1.1*max(self.freq))
            self.ax.set_ylim(1.1*min(self.ft_vals), 1.1*max(self.ft_vals))


    def create_ax(self):
        """
        Routine to initialize the ax object depending of the display state
        """
        lim = 2
        if self.state.is_3d():
            self.ax = self.fig.add_subplot(projection = "3d")
            self.ax.set_xlim(-lim, lim)
            self.ax.set_ylim(-lim, lim)
            self.ax.set_zlim(-lim, lim)
            if self.state.state == "spins":
                self.artist = self.ax.quiver(*self.spins.to_quiver())
                self.atoms, = self.ax.plot(self.spins.x, self.spins.y, self.spins.z, linestyle="", marker="o", color = "gray")
            if self.state.state == "mag":
                self.artist = self.ax.quiver(*self.M.magnetization())
                self.mag_trace, = self.ax.plot([m[0] for m in self.mag_vals], [m[1] for m in self.mag_vals], [m[2] for m in self.mag_vals], color="green", linewidth = "0.5")
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim(self.spins._pars["start_t"], self.spins._pars["end_t"])
            self.ax.set_ylim(-2.2, 2.2)
            if self.state.state == "fid":
                self.artist, = self.ax.plot(self.t_arr, self.fid_arr)
                self.ax.set_xlim(self.sim._pars["start_t"], self.sim._pars["end_t"])
                self.ax.set_xlabel("Temps (s)")
                self.ax.set_ylabel("Amplitude (a.u.)")
            if self.state.state == "ft":
                self.artist, = self.ax.plot(self.freq, self.ft_vals)
                self.ax.set_xlabel("Déplacement chimique (ppm)")
                self.ax.set_ylabel("Amplitude (a.u.)")
        self.ax.set_title(self.state.state)


    def ft(self):
        """
        Computes the Fourier transform of the fid just like in NMR processing softwares
        """
        window = np.array([self.wm(t) for t in self.t_arr])
        self.fid_arr = window * self.fid_arr
        self.freq = rfftfreq(len(self.t_arr), self.sim._pars["timestep"])
        self.ft_vals = np.abs(rfft(self.fid_arr))
