import numpy as np 
from OrrallZirkerPy.Atmos import AmbientPops, SuprathermalParticles 
from OrrallZirkerPy.AtomData import CSecActive
from OrrallZirkerPy.SuprathermalPops import CalcPops
import copy 
import sys

"""
Main routine, that calculates the non-thermal radiation produced
from impact of non-thermal ions with ambient plasma. 

Charge exchange, and subsequent redistribution of electrons through
collisions (radiative excitation is much slower process and not 
considered here) transforms some fraction of the suprathermal ion 
beam into a suprathermal neutral beam (or less ionised, in the case 
of species other than H). That netural beam can then emit radiation, 
which is heavily doppler shifted and appears in the wing of the line.

-- The atomic data used for this is obtained via AtomData, which collates the 
required cross sections. The recommended (by us) cross sections are 
hard coded in AtomData.CSecActive, but are straightforward to swap for others, 
so long as they are contained with the CrossSections.py routine. 
It is necessary to fit some function to underlying data to use cross sections 
in this package.

-- The atmosphere data used is held via Atmos.py, though ultimately is user 
provided via OZ() inputs. While this package was written with RADYN+FP
in mind, any suitable atmosphere may be input, you just need the ambient 
electron, hydrogen, and proton density into which the particle beam has been
injected. This also stores the ion beam data. This does assume a format similar
to RADYN+FP, so please check Atmos.py carefully to ensure you are providing the 
correct data. Data can be input for a 
    * single point in depth at one snapshot in time, 
    * single point in depth at multiple times
    * array of depth points at one snapshot in time
    * array of depth points at multiple times. 
Realistically, the latter two are necessary to appreciate the full emission. 
Similarly, a single ion energy can be input, or the full energy distribution, but
again, to appreciate the full emission you should input a range of energies.

-- The Cross Section, Atmosphere, and Non-thermal particles objects are created from
input provided by the user, and passed to SuprathermalPops.py, which solves the statistical 
equilibrium equations and returns the populations of the atomic levels requrested. 

-- This object is then passed to Emission.py, which computes as a function of energy, 
i.e. ion velocity, the emissivity [erg/s/cm^3/sr/Ang]. The emissivity is equivalent to 
the contribution fn. Methods are inlucuded in that class to compute the intensity in each 
cell (the emissivity * dz), and the full height-integrated emission (assuming a vertical loop), 
on both energy and the equivalent wavelength scale (in Angstrom relative to line core).


NOTES
_____

** Currently only set up for Hydrogen 2 or 3 level atom (Lyman alpa, Lyman 
beta, Balmer alpha). Extending to arbritary species will take some reorganising. 
... 31st July 2021, Graham Kerr

** Future plans are to include Helium
... 31st July 2021, Graham Kerr


References
__________

Some relevant references are: 


References related to cross sections are found in CrossSections.py

"""


def OZ(nLev=3, 
	   nHyd=0, nElec=0, nProt=0, height=0, times=0, 
	   nthmp_e=0, nthmp_mu=0, nthmp_f=0, ionfrac=1.0,
	   isum = -1 
	   ):
    """
    Compute the non-thermal emission associated with the presence of a 
    suprathermal ion beam distribution in a solar/stellar atmosphere (e.g
    during solar flares). This is the Orrall-Zirker effect.

    Inputs
    ______

    REQUIRED:
    nHyd -- float
            Number density of ambient hydrogen atoms in the target atmosphere
            Can be defined as an array in height and/or time, or a single value. In cm^-3.
    nElec -- float
            Number density of ambient electrons in the target atmosphere
            Can be defined as an array in height and/or time, or a single value. In cm^-3.
    nProt -- float
            Number density of ambient protons in the target atmosphere
            Can be defined as an array in height and/or time, or a single value. In cm^-3.
    nthmp_e -- float
            Energies at which to evaluate the OZ effect. These are the energies
            at which the injected suprathermal particle distribution are defined.
            Can be array or single value. In keV.
    nthmp_f -- float
            The injected suprathermal particle distribution.
            Can be defined as an array in height and/or time, or a single value. 
            In particles/cm^3/sr/keV.
   
    OPTIONAL:
    nLev -- int
            Number of bound levels of the required species to solve 
            Optional, default = 3 (including the continuum level gives
            nLev+1)
    height -- float
            Height scale of the target atmosphere
            Optional (though required to compute intensity). 
            Can be defined as an array in height and/or time, or a single value. In cm.
    times -- float
            Time in the simulation
            Optional (only really used to bundle with the output to keep
            everything together). 
            Can be array or single value. In s.
    nthmp_mu -- float
            Pitch angle of the injected suprathermal particle distribution, if that
            is resolved. 
            Optional (only required is pitch angle is resolved). 
    ionfract -- float
             The fraction of the injected particle spectrum (nthmp_f) that is the ion of 
             interest. Optional, default = 1.0 (pure proton beam).
    isum -- int
            The level to be replaced by particle conservation equation. For most every purpose
            this is the nLev+1 (i.e proton/ion density), or '-1'. Be careful changing this. 
            Optional, default = -1

    Outputs
    _______
    
    ***** INCLUDE DISCUSSION OF OUTPUTS ****

    Notes
    ______

    Graham Kerr
    August 2021


    """
    
    ## Some param checks
    if (nLev != 2) and (nLev !=3):
    	print('\n>>> You have not entered a valid value of nLev (nLev = 2 or 3)\n     Defaulting to nLev = 3!')
    	nLev = 3

    ## These are must-have variables. The rest are optional 
    if ((len(np.array(nHyd).shape) == 0 and nHyd == 0) 
    	or (len(np.array(nProt).shape) == 0 and nProt == 0) 
        or (len(np.array(nElec).shape) == 0 and nElec == 0) 
        or (len(np.array(nthmp_e).shape) == 0 and nthmp_e == 0) 
        or (len(np.array(nthmp_f).shape) == 0 and nthmp_f == 0)):
    	print('\n>>> You have not included all of the required inputs. The following are all needed:')
    	print('nHyd (number density of ambient Hydrogen atoms)')
    	print('nElec (number density of ambient electrons)')
    	print('nProt (number density of ambient protons)')
    	print('nthmp_e (energy at which which to evaluate non-thermal pops)')
    	print('nthmp_f (non-thermal proton distribution fn in particles/cm^3/sr/keV)')
    	sys.exit('Exiting... \nEnter all the above inputs to correct')    

    energy = copy.deepcopy(nthmp_e)

    ## Create the atmosphere object
    atmos = AmbientPops(nHyd=nHyd, nElec=nElec, nProt=nProt, height=height, times=times)

    # ## Create the Non-thermal proton distribution object
    nthmp = SuprathermalParticles(nthmp_e=nthmp_e, nthmp_mu=nthmp_mu, nthmp_f=nthmp_f, ionfrac=ionfrac)

    ## Create the cross secion object
    csecA = CSecActive(energy, nLev=nLev)

  
    ## Add additional dimnesions to atmosphere object if necessary
    if len(atmos.nElec.shape) == 2:
        nDim1 = atmos.nElec.shape[0]
        nDim2 = atmos.nElec.shape[1]
        if (nDim1 != nthmp.fe.shape[0]) or (nDim2 != nthmp.fe.shape[1]):
        	sys.exit('\n>>> Exiting... \nDimenions of ambient particles dont match dimensions of injected proton spectrum.\nCheck your depth and time grid\n')
    if len(atmos.nElec.shape) == 1:
    	nDim1 = atmos.nElec.shape[0]
    	nDim2 = 1
    	atmos.nProt = np.repeat(atmos.nProt[:, np.newaxis],1, axis=1)
    	atmos.nHyd = np.repeat(atmos.nHyd[:, np.newaxis],1, axis=1)
    	atmos.nElec = np.repeat(atmos.nElec[:, np.newaxis],1, axis=1)
    	atmos.height = np.repeat(atmos.height[:, np.newaxis],1, axis=1)
    	# atmos.times = np.repeat(atmos.times[:, np.newaxis],1, axis=1)
    	nthmp.fe = np.repeat(nthmp.fe[:, np.newaxis,:],1, axis=1)
    	if (nDim1 != nthmp.fe.shape[0]) or (nDim2 != nthmp.fe.shape[1]):
    		sys.exit('\n>>> Exiting... \nDimenions of ambient particles dont match dimensions of injected proton spectrum.\nCheck your depth and time grid\n')
    if len(atmos.nElec.shape) == 0:
        nDim1 = 1
        nDim2 = 1
        atmos.nProt = np.repeat(atmos.nProt[np.newaxis,np.newaxis],1, axis=0)
        atmos.nHyd = np.repeat(atmos.nHyd[np.newaxis,np.newaxis],1, axis=0)
        atmos.nElec = np.repeat(atmos.nElec[np.newaxis,np.newaxis],1, axis=0)
        atmos.height = np.repeat(atmos.height[np.newaxis,np.newaxis],1, axis=0)
        # atmos.times = np.repeat(atmos.times[np.newaxis,np.newaxis],1, axis=0)
        nthmp.fe = np.repeat(nthmp.fe[np.newaxis, np.newaxis,:],1, axis=0)
        if (nDim1 != nthmp.fe.shape[0]) or (nDim2 != nthmp.fe.shape[1]):
        	sys.exit('\n>>> Exiting... \nDimenions of ambient particles dont match dimensions of injected proton spectrum.\nCheck your depth and time grid\n')
    if len(atmos.nElec.shape) != len(nthmp.fe.shape)-1:
        sys.exit('\n>>> Exiting... \nDimenions of ambient particles dont match dimensions of injected proton spectrum.\nCheck your depth, time, and energy grids\n')


    SupraThmPops = CalcPops(csecA, atmos, nthmp, isum=isum)

    # emiss = CalcEmissiv(SupraThmPops)

    return SupraThmPops

