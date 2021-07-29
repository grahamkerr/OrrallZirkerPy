import numpy as np
from scipy import constants


def energy2vel(energy, particle = 'proton'):
    """
     
    A function to return the particle velocity in cm/s, given an 
    energy in keV.
   
    Inputs
    _______

    energy -- [float] 
              Projectile energy in keV.
              Can be either a single value, or an array.
    particle -- [string]
                Projectile type
                Currently accepts 'proton' or 'electron'
                [optional, default is 'proton']

    Outputs
    _______

    velocity -- [float]
                Projectile velocity in cm/s

    NOTES
    ______

    Assumes projectile is a proton. In the formulation of the Orrall-Zirker 
    effect we assume the speed of the energetic neutral hydrogen particle is
    the same as the proton from which is was created via charge exchange. 

    Graham Kerr
    July 2021

    """
    ########################################################################
    # Some preliminary set up
    ########################################################################

    ## Turn to np array if an integer or float are provided
    if type(energy) == int:
        energy = np.array(energy)
    if type(energy) == float:
        energy = np.array(energy)
    if type(energy) == tuple:
        energy = np.array(energy)
 
    ## Some constants
    clight = constants.c*1e2 # cm/s
    if particle == 'proton':
        E_p = constants.value('proton mass energy equivalent in MeV')*1e3 #MeV/c^2
    elif particle == 'electron':
        E_p = constants.value('electron mass energy equivalent in MeV')*1e3 #MeV/c^2
    else:
    	print('\n>>> You have not entered a valid string ("proton" or "electron")\n     Defaulting to proton')
    	E_p = constants.value('proton mass energy equivalent in MeV')*1e3 #MeV/c^2

   

    ########################################################################
    # Calculate the velocity
    ########################################################################

    velocity = clight*np.sqrt(1.00 - ((E_p)/(energy+E_p))**2)

    return velocity