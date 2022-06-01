import numpy as np 
import OrrallZirkerPy as OZpy
from OrrallZirkerPy.EnergyToVel import energy2vel 
from OrrallZirkerPy.AtomData import Transitions
from scipy import constants 
import sys
import copy 

class CalcEmissiv:
    """
    Computes the emission as a function of energy (and for space, time if 
    multi-dimensional). 

    CalcEmissiv returns the emissivity in each cell, in units of
    erg/s/cm^2/sr/Ang/cm

    Inputs
    ______

    Nthm_data: an object with the following entries 
    * Nthm_pops -- the nonthermal population densities as a function of proton 
	     		   energy [and optionally in space and time]. Particles /cm^3.
		    	   dimensions are [space, time, nLev+1, nLev+1, energy]. Even if space
			       or time isn't considered those dimnensions are included, of size one.
    * nLev -- the number of level in the model atom under consideration (default
              is nLev 3). Should match the dimensions if Nthm_pops. 
    * Nthm_en -- the energy at which the pops are defined in keV

    
    Ouputs
    ______

    An object containing

    * emiss -- The emissivity at each spatial and temporal pixel, at each energy considered.
	     	 [space, time, nTrans, energy], where nTrans is the number of transitions 
		     considered. The data for the transitions is held within AtomData.Tramsitions
    * dLambda -- The shift in wavelength from line center, in angstroms
    * wavelength_rest -- The rest wavelength


    
    Methods
    _______

    emiss2int_total(height): A fn to compute the emergent intensity, given the height array. 

    emiss2int_z(height): A fn to compute the intensity in each grid cell, given the height array. 


    Notes
    _____


 	Graham Kerr
    August 2021

    """
	

    def __init__(self,Nthm_data):
	    
    

        ########################################################################
        # Some preliminary set up
        ########################################################################

        ### determine dimensions
        nDim1 = Nthm_data.NPops.shape[0]
        nDim2 = Nthm_data.NPops.shape[1]
        nDimE = Nthm_data.NPops.shape[3]
    
        ### Grab the atomic data regarding the transitions
        trans = Transitions(nLev = Nthm_data.nLev, species=Nthm_data.species)
        nDimTr = trans.Aji.size

        ### Do all dimensions match (this should be rare if running as part of the 
        ### main package... though might creep in if using separately)
        if nDimE != Nthm_data.energy.shape[0]:
            sys.exit('>>> Exiting... \nThere is a mismatch with energy dimensions in Nthm_data') 
        if Nthm_data.nLev+1 != Nthm_data.NPops.shape[2]:
            sys.exit('>>> Exiting... \nThere is a mismatch with number of levels in Nthm_data') 

        emiss = np.zeros([nDim1, nDim2, nDimE, nDimTr], dtype = np.float64)

    ########################################################################
    # Calculate the emissivity
    ########################################################################
        if Nthm_data.species == 'H':
            if nDimTr > 1:

                val1 = (2.0*trans.mass)**0.5
                val2 = np.array(4.0*constants.pi*trans.wavelength_rest)
        
                for eind in range(nDimE):
                    for trind in range(nDimTr):
                        emiss[:, :, eind, trind] = (val1/val2[trind] * 
            		                         Nthm_data.energy[eind]**0.5 * 
            		                         Nthm_data.NPops[:,:,trans.upplev[trind],eind] * 
            		                         trans.Aji[trind] ) * trans.phot2erg[trind]


                dVel = energy2vel(Nthm_data.energy,particle='proton')/1e5
                dLambda = np.zeros([nDimE, nDimTr], dtype = np.float64)
                for trind in range(nDimTr):
                    dLambda[:,trind] = dVel/(constants.c/1e3) * trans.wavelength_rest[trind]

            elif nDimTr == 1:

                val1 = (2.0*trans.mass)**0.5
                val2 = np.array(4.0*constants.pi*trans.wavelength_rest)

                for eind in range(nDimE):
                    emiss[:, :, eind, 0] = (val1/val2 * 
                                             Nthm_data.energy[eind]**0.5 * 
                                             Nthm_data.NPops[:,:,trans.upplev,eind] * 
                                             trans.Aji ) * trans.phot2erg
                    dVel = energy2vel(Nthm_data.energy,particle='proton')/1e5
                    dLambda = np.zeros([nDimE, nDimTr], dtype = np.float64)
                    dLambda[:,0] = dVel/(constants.c/1e3) * trans.wavelength_rest

        elif Nthm_data.species == 'He':
            if nDimTr == 1:
               
                val1 = (2.0*trans.mass)**0.5
                val2 = np.array(4.0*constants.pi*trans.wavelength_rest)

                for eind in range(nDimE):
                    emiss[:, :, eind, 0] = (val1/val2 * 
                                             Nthm_data.energy[eind]**0.5 * 
                                             Nthm_data.NPops_HeIIex[:,:,eind] * 
                                             trans.Aji ) * trans.phot2erg
                    dVel = energy2vel(Nthm_data.energy,particle='alpha')/1e5
                    dLambda = np.zeros([nDimE, nDimTr], dtype = np.float64)
                    dLambda[:,0] = dVel/(constants.c/1e3) * trans.wavelength_rest

        self.emiss = emiss
        self.wavelength_rest = trans.wavelength_rest
        self.energy = Nthm_data.energy
        self.dVel = dVel
        self.dLambda = dLambda
        self.units = 'Emissivity (emiss) in erg/s/cm^2/sr/A/cm; wavelengths in angstroms; Energy in keV; velocity in km/s'

           
       
    def emiss2int_z(self, height, harr = 1):
        """
    	A function to compute the intensity in each cell, given the height array.

    	In each cell the intensity is emissivity * dZ

    	Inputs
    	______

    	height -- the height array in cm. There should be a height array for each 
    	          dimension of self.emiss. 
    	harr -- which index of emiss coresponds to height (default is '1')

    	Outputs
    	_______

    	intensity_z -- the intensity in each grid cell for each transition 
    	               in erg/s/cm^2/sr/Ang. 

    	Notes
    	______

    	* Some checking of array dimensions is performed, but be careful here.
    	  Make sure you pass a height array that matches the dimensions of 
    	  emiss, since time and height can be either one of the first two 
    	  indices (depending on your input earlier).

    	Graham Kerr
    	August 2021

        """
        if len(height.shape) == 2:
            nDim1 = height.shape[0]
            nDim2 = height.shape[1]
            if (nDim1 != self.emiss.shape[0]) or (nDim2 != self.emiss.shape[1]):
        	    sys.exit('\n>>> Exiting... \nDimenions of height array dont match dimensions of emission array.\nCheck your depth and time grid\n')
        if len(height.shape) == 1:
    	    nDim1 = height.shape[0]
    	    nDim2 = 1
    	    height = np.repeat(height[:, np.newaxis],1, axis=1)
    	    harr = 0
    	    if (nDim1 != self.emiss.shape[0]) or (nDim2 != self.emiss.shape[1]):
        	    sys.exit('\n>>> Exiting... \nDimenions of height array dont match dimensions of emission array.\nCheck your depth and time grid\n')
        if len(height.shape) == 0:
            nDim1 = 1
            nDim2 = 1
            height = np.repeat(height[np.newaxis,np.newaxis],1, axis=0)
            if (nDim1 != self.emiss.shape[0]) or (nDim2 != self.emiss.shape[1]):
        	    sys.exit('\n>>> Exiting... \nDimenions of height array dont match dimensions of emission array.\nCheck your depth and time grid\n')
        if nDim1 == 1 and nDim2 == 1:
        	sys.exit('>>> Exiting... \nYou must enter a height array, not a single value, to evaluate the intensity in each cell (you need a dZ)')
        if (harr != 0 and harr !=1):
            harr = 1
            print(">>> harr must be '0', or '1'... setting to default value of 1")

        nDim1_emiss = self.emiss.shape[0]
        nDim2_emiss = self.emiss.shape[1]
        nDimE = self.energy.shape[0]
        nDimTr = self.wavelength_rest.size

        intensity_z = copy.deepcopy(self.emiss)

        if harr == 1:
            ### Difference in height (top of atmosphere is set to 0)
            ### Finds where the largest height is (top of atmosphere) 
            ind = np.where(height[0,:] == np.max(height[0,:]))[0][0]
            dZ = np.roll(height[:,:],1) - height[:,:] 
            dZ[:,ind]= 0
        elif harr == 0:
        	### Difference in height (top of atmosphere is set to 0)
            ### Finds where the largest height is (top of atmosphere) 
            ind = np.where(height[:,0] == np.max(height[:,0]))[0][0]
            dZ = np.roll(height[:,:],1) - height[:,:] 
            dZ[ind,:]= 0

        if nDimTr > 1:
            for trind in range(nDimTr):
                for eind in range(nDimE):
                    intensity_z[:,:,eind,trind]*=dZ
        elif nDimTr == 1:
            for eind in range(nDimE):
                intensity_z[:,:,eind,0]*=dZ

        return intensity_z

    def emiss2int_total(self, height, harr=1):
        """
    	A function to compute the emergent intensity summed over the loop, given the height array.

    	The intensity is integral(emissivity * dZ)

    	Inputs
    	______

    	height -- the height array in cm. There should be a height array for each 
    	          dimension of self.emiss. 
    	harr -- which index of emiss coresponds to height (default is '1')

    	Outputs
    	_______

    	intensity_total -- the intensity summed through the loop for each transition 
    	                   in erg/s/cm^2/sr/Ang. 

    	Notes
    	______

    	* Some checking of array dimensions is performed, but be careful here.
    	  Make sure you pass a height array that matches the dimensions of 
    	  emiss, since time and height can be either one of the first two 
    	  indices (depending on your input earlier).

    	Graham Kerr
    	August 2021
        """

        if len(height.shape) == 2:
            nDim1 = height.shape[0]
            nDim2 = height.shape[1]
            if (nDim1 != self.emiss.shape[0]) or (nDim2 != self.emiss.shape[1]):
        	    sys.exit('\n>>> Exiting... \nDimenions of height array dont match dimensions of emission array.\nCheck your depth and time grid\n')
        if len(height.shape) == 1:
    	    nDim1 = height.shape[0]
    	    nDim2 = 1
    	    height = np.repeat(height[:, np.newaxis],1, axis=1)
    	    harr = 0
    	    if (nDim1 != self.emiss.shape[0]) or (nDim2 != self.emiss.shape[1]):
        	    sys.exit('\n>>> Exiting... \nDimenions of height array dont match dimensions of emission array.\nCheck your depth and time grid\n')
        if len(height.shape) == 0:
            nDim1 = 1
            nDim2 = 1
            height = np.repeat(height[np.newaxis,np.newaxis],1, axis=0)
            if (nDim1 != self.emiss.shape[0]) or (nDim2 != self.emiss.shape[1]):
        	    sys.exit('\n>>> Exiting... \nDimenions of height array dont match dimensions of emission array.\nCheck your depth and time grid\n')
        if nDim1 == 1 and nDim2 == 1:
        	sys.exit('>>> Exiting... \nYou must enter a height array, not a single value, to evaluate the intensity in each cell (you need a dZ)')
        
        if (harr != 0 and harr !=1):
            harr = 1
            print(">>> harr must be '0', or '1'... setting to default value of 1")

        nDim1_emiss = self.emiss.shape[0]
        nDim2_emiss = self.emiss.shape[1]
        nDimE = self.energy.shape[0]
        nDimTr = self.wavelength_rest.size

    	
        if (harr == 1):
            intensity_tot = np.zeros([nDim1, nDimE, nDimTr], dtype=np.float64)
        elif (harr == 0):
        	intensity_tot = np.zeros([nDim2, nDimE, nDimTr], dtype=np.float64)

        if nDimTr > 1:
            for trind in range(nDimTr):
                for eind in range(nDimE):
                    intensity_tot[:,eind,trind] = np.trapz(np.flip(self.emiss[:,:,eind,trind],axis=harr), x = np.flip(height[:,:],axis=harr),axis=harr)
        elif nDimTr == 1:
            for eind in range(nDimE):
                intensity_tot[:,eind,0] = np.trapz(np.flip(self.emiss[:,:,eind,0],axis=harr), x = np.flip(height[:,:],axis=harr),axis=harr)

        return intensity_tot
