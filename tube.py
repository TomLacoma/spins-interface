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
import math

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

        self.base_quiver = self.set_base(-2,-2,-3,0.5) #Computes base vectors to plot

        self.t_arr, self.fid_arr = [spins._pars["start_t"]], [M.fid] #time and fid storing arrays
        self.freq = np.arange(0,15,0.005) #freq and FT storage arrays
        self.ft_vals = [0 for i in self.freq] 
        self.mag_vals=[] #values taken by magnetization vector

        self.wm = lambda t:np.exp(-t*0.3) #exponential, 0.3Hz line broadening

        self.fig = Figure() #figure environment embedded in tk
        
        #Created a tkinter window
        self.root = tk.Tk()
        self.root.configure(bg='white')
        self.root.wm_title("SPINS v0.2")

        self.mainloop = self.root.mainloop
        #Creates a canva for embedding the matplotlib mess
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # A tk.DrawingArea.
        self.canvas.draw()

        #Creates a frame for the title
        self.f_title = tk.Frame(self.root)
        self.title=tk.Label(self.f_title, text=f'{self.state.st_dict[self.state.state]}', font=("Calibri", 15), bg="white")
        self.title.pack(side='top', fill='both', expand='True')
        #Creates a frame to contain the UI buttons
        self.f_buttons = tk.Frame(self.root)

        #Creates the UI buttons...
        self.next_btn = tk.Button(master=self.f_buttons, text="Next", command=self.next_fig)
        self.switch_btn = tk.Button(master=self.f_buttons, text = "Acquire !", command = self.pulse)
        self.prev_btn = tk.Button(master=self.f_buttons, text="Previous", command=self.prev_fig)
        self.lift_btn = tk.Button(master=self.f_buttons, text="Lift sample", command=self.toggle_lift)
        #...and add them to the frame
        self.prev_btn.pack(side=tk.LEFT, expand=True)
        self.switch_btn.pack(side=tk.LEFT, expand=True)
        self.lift_btn.pack(side=tk.LEFT, expand=True)
        self.next_btn.pack(side=tk.LEFT, expand=True)

        #Organize the canva with matplotlib display and the frame 
        self.f_title.pack(side='top', fill='both', pady=5)
        self.canvas.get_tk_widget().pack(side="top", fill='both', expand=True)
        self.f_buttons.pack(side='top', fill='both', pady=5)

        #Animation function for the matplotlib animation, crucial
        self.ani = FuncAnimation(self.fig, self.update, frames = sim.frames, interval = spins._pars["timestep"], cache_frame_data=False)
        #Handles the windows clising event preventing softlocks
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
        #Different mpl artists for different purposes
        self.ax, self.artist, self.atoms, self.mag_trace, self.B0 = None, None, None, None, None
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
        self.ax.set_facecolor("black")
        self.sim.flashed = True


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
            #Resets the saved time, fid and magnetization values
            self.t_arr = []
            self.fid_arr = []
            self.mag_vals = []

        if self.sim.flashed:
            self.ax.set_facecolor("white")
            self.sim.flashed = False

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

        #Sets the graph title
        self.title.config(text=self.state.title())
        #Updates all of the physics
        self.M.update(t)
        self.spins.update(t)
        self.t_arr.append(self.sim.curr_t)
        self.fid_arr.append(self.M.fid)
        self.mag_vals.append(self.M.magnet)

        #Big array of conditions plotting the rigth values depending of the disp state
        if self.state.state == "spins":
            if not self.atoms:
                self.atoms, = self.ax.plot(self.spins.x, self.spins.y, self.spins.z, linestyle="", marker="o", color = "gray", zorder=1)
            self.artist = self.ax.quiver(*self.spins.to_quiver(), pivot='middle', color=self.spins.quiver_color, zorder=0)
        if self.state.state == "energy":
            self.atoms.set_data_3d(self.spins.ex, self.spins.ey, self.spins.ez)
            self.artist = self.ax.quiver(*self.spins.to_energy_quiver(), pivot='middle', color=self.spins.quiver_color, zorder=0)
        if self.state.state == "mag":
            self.artist = self.ax.quiver(*self.M.magnetization(), linewidth = 2)
            self.mag_trace.set_data_3d([m[0] for m in self.mag_vals], [m[1] for m in self.mag_vals], [m[2] for m in self.mag_vals])
        if self.state.state == "fid": 
            self.artist.set_data(self.t_arr, self.fid_arr)
            self.ax.set_xlim(self.sim._pars["start_t"], self.sim._pars["end_t"])
        if self.state.state == "ft":
            pass


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
                self.ax.computed_zorder = False
                self.artist = self.ax.quiver(*self.spins.to_quiver())
                self.atoms, = self.ax.plot(self.spins.x, self.spins.y, self.spins.z, linestyle="", marker="o", color = "gray")
                self.B0 = self.ax.quiver(-2,-2,-1,0,0,2, color = "darkslategrey")
                txt = self.ax.text(-2, -2, 1.5, "$\\vec B_0$", color='darkslategrey')
                self.trig = self.ax.quiver(*self.base_quiver, color=["red","green","blue"])
            if self.state.state == "energy":
                lim = 3
                self.ax.set_xlim(-lim, lim)
                self.ax.set_ylim(-lim, lim)
                self.ax.set_zlim(-lim, lim)
                self.ax.computed_zorder = False
                self.ax.view_init(elev=0, azim=-90, roll=0)
                self.artist = self.ax.quiver(*self.spins.to_energy_quiver())
                self.atoms, = self.ax.plot(self.spins.ex, self.spins.ey, self.spins.ez, linestyle="", marker="o", color = "gray", zorder=1)
                txt = self.ax.text(-4, 0, 1, "$+\\frac{E_0}{2}$", color='darkslategrey')
                txt2 = self.ax.text(-4, 0, -1, "$-\\frac{E_0}{2}$", color='darkslategrey')
            if self.state.state == "mag":
                self.artist = self.ax.quiver(*self.M.magnetization())
                self.mag_trace, = self.ax.plot([m[0] for m in self.mag_vals], [m[1] for m in self.mag_vals], [m[2] for m in self.mag_vals], color="green", linewidth = "0.5")
                txt = self.ax.text(-2, -2, 1, "$\\vec B_0 $", color='darkslategrey')
                self.B0 = self.ax.quiver(-2,-2,-0.5,0,0,1, color = "darkslategrey")
                self.trig = self.ax.quiver(*self.base_quiver, color=["red","green","blue"])
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
                self.artist.set_data(self.freq, [self.spectrum(f, 0.01) for f in self.freq])
                self.ax.set_xlim(15, 0)
                self.ax.set_ylim(-.5, 1.5*max([self.spectrum(f, 0.01) for f in self.freq]))

    def ft(self):
        """
        Computes the Fourier transform based on the content of the fid list
        """
        window = np.array([self.wm(t) for t in self.t_arr])
        self.fid_arr = window * self.fid_arr
        self.freq = np.arange(0,15,0.01)
        self.ft_vals = np.abs(rfft(self.fid_arr))
    
    def spectrum(self, d, std):
        """Computes the spectrum based upon the peak list of Spins object, a list of [chemical shift, multiplicity, coupling constant] items at chemical shift d"""
        res = 0
        for mult in self.spins.mana:
            if mult[1] <= 1:
                res += gaussian(d, mult[0], std)
            k = int(mult[1]//2)
            if mult[1]%2 ==0:
                peak = np.array(list(range(-k, k))) + 0.5
            else:
                peak = np.array(list(range(-k, k+1)))
            #Mults now contains the offsets for chemical shifts in multiplet as multiples of J
            for i,m in enumerate(peak):
                res += math.comb(int(mult[1])-1, i)*gaussian(d, mult[0]+mult[2]*m, std)
            return res

    def set_base(self, x, y, z, mag):
        """Computes the values to display base vectorx x,y and z with magnitude mag"""
        return [x,y,z,[mag, 0, 0],[0,mag,0],[0,0,mag]]

def gaussian(x, avg, std):
    """Normalized gaussian function"""
    return(np.exp(-0.5*((x-avg)/std)**2)/(std*np.sqrt(2*np.pi)))