import numpy as np 
import OrrallZirkerPy as OZpy
from OrrallZirkerPy.EnergyToVel import energy2vel 
from OrrallZirkerPy.AtomData import Transitions
# from OrrallZirkerPy.Atmos import AmbientPops, SuprathermalProtons 
from scipy import constants 


"""
Computes the emission as a function of energy (and for space, time if 
multi-dimensional). 

CalcEmissiv returns the emissivity in each cell, in units of
erg/s/cm^2/sr/Ang/cm

"""
class CalcEmissiv:

	"""
	Calculate the emissivity of nonthermal emission given population densities
	of the levels under consideration.

	Inputs
	______

	Nthm_data 
	* Nthm_pops -- the nonthermal population densities as a function of proton 
				   energy [and optionally in space and time]. Particles /cm^3.
				   dimensions are [space, time, nLev+1, nLev+1, energy]. If space
				   or time isn't considered those dimnensions are omitted.
	* nLev -- the number of level in the model atom under consideration (default
	          is nLev 3). Should match the dimensions if Nthm_pops. 
	* Nthm_en -- the energy at which the pops are defined in keV

	Ouputs
	______

	An object containing

	emiss -- The emissivity at each spatial and temporal pixel, at each energy considered.
			 [space, time, nTrans, energy], where nTrans is the number of transitions 
			 considered. The data for the transitions is held within AtomData.Tramsitions
	deltaL -- The shift in wavelength from line center, in angstroms

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

    
    ### Grab the atomic data regarding the transitions
		trans = Transitions(nLev = nLev)


    ### Do all dimensions match


		emiss = np.zeros([nDim1, nDim2, nDimE, nDimTr], dtype = np.float64)

    ########################################################################
    # Calculate the emissivity
    ########################################################################
   
		val1 = (2.0*trans.mass)**0.5
		val2 = 4.0*constants.pi*trans.wavelength_rest
    
		for eind in range(len(Nthm_data.energy)):
        
			for trind in range(len(trans.wavelength_rest)):
				emiss[:, :, eind, trind] = (val1/val2[trind] * 
        		                         Nthm_data.energy[eind]**0.5 * 
        		                         Nthm_data.pops[:,:,trans.upplev[trind],eind] * 
        		                         trans.Aji[trind] ) * trans.phot2erg[trind]

        # class emiss_out:
        #     def __init__(selfout):
		self.emiss = emiss
		self.wavelength_rest = wavelength_rest
		self.energy = Nthm_data.energy
		self.dVel = energy2vel(Nthm_data.energy)
		self.dLambda = self.dVel/(constants.c/1e3)*self.wavelength_rest

            # selfout.height = height
            # selfout.times = times
            # selfout.Units = 'velocity in [km/s], Pops in [particles cm^-3 keV^-1]'

    # out = SupraThermPops_out()

       
		def emiss2int_z(self, atmos):

			intensity_z = 0

			return intensity_z

		def emiss2int_total(self, atmos):

			intensity_total = 0
        
			return intensity_total

