from spins import *
from tube import * 

lim = 2 #Global limit for plots
random.seed() #Sets a new random seed
pos = np.linspace(-1,1,3) #Array of positions for spins (cube)

#The pars parameters below can be played with
frame_pars = {"start_t":0, "end_t":5, "timestep":1/60} #Simulation parameters
pars = {"angle":20, "mag":1, "T1":0.5, "T2":1, "f":2, "atom_size":0.1} #Physics parameters 
#Alright enough playing

sim = Sim(pars = pars, frame_pars=frame_pars) #Creates a simulation environment
spins = Spins(pos, pos, pos, sim, phase = lambda x:2*np.pi*random.random()) #creates spins object

sim._pars["TD0"] = 1 #Manually sets the acquisition loop counter to 1 

zg = R.from_euler('x', np.pi/2) #Pulse program : pi/2 impulsion along x

M = Magnetization(sim, pulseprog = zg) #Instance of Magnetization object, handles bulk magnetization
state = DispState("spins") #Instance of DispState handling the Rightful Displaying of Things

tube = Tube(sim, spins, M, state)
tube.show_axis = False

#Keyboard control
tube.root.bind("<space>", lambda x:tube.pulse())
tube.root.bind("<Left>", lambda x:tube.prev_fig())
tube.root.bind("<Right>", lambda x:tube.next_fig())
tube.root.bind("<Up>", lambda x: tube.lift())
tube.root.bind("<Down>", lambda x: tube.unlift())
tube.root.bind('<Escape>', lambda x: tube.cleanup())

#Here we go !
tube.mainloop()