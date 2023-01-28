import numpy as np
from scipy import constants 
import sys
"""

This module contains two classes to hold the atmospheric properties.

AmbientPops holds the population densities (in cm^-3) for ambient particles:
hydrogen, protons, electrons. Also stored are the height(s) and time(s) of 
the input data.

SuprathermalProtons holds the injected non-thermal proton distribution, from 
which the suprathermal neutrals will be calculated. Associated products are 
stored there also. 

See the individual classes for the requirements. 

Graham Kerr
August 2021

"""

class AmbientPops:
    """
     
    Holds the ambient number densities of neutral hydrogen,
    electrons, and protons. These particle densities are used
    along with the collisional cross sections to compute the 
    populations of suprathermal particles.
   
    Inputs
    _______

    nHyd -- Hydrogen number density [cm^-3]
    nElec -- Electron number density [cm^-3]
    nProt -- Proton number density [cm^-3]
    height -- Height in [cm]
    time  -- Simulation time in seconds

    These can be either 
    1) single values (i.e. one depth point at one time),
    2) 1D array of values (i.e. many depth points at one time)
    3) 2D array of values where multiple depth points are defined
       at multiple times. 
    However, all three particle densities should have the same dimensions,
    and a warning message will display if this is not the case.

    Outputs
    _______

    An object with each population density in cm^-3

    Graham Kerr
    July 2021

    """
	
    def __init__(self, nHyd=0, nElec=0, nProt=0, height = 0, times = 0):
    
    ########################################################################
    # Some preliminary set up
    ########################################################################

        ###
        ### Check the dimensions
        ###
        ## Turn to np array if an integer or float are provided
        if type(nHyd) == int:
            nHyd = np.array(nHyd)
        if type(nHyd) == float:
            nHyd = np.array(nHyd)
        if type(nHyd) == tuple:
            nHyd = np.array(nHyd)
        if type(nProt) == int:
            nProt = np.array(nProt)
        if type(nProt) == float:
            nProt = np.array(nProt)
        if type(nProt) == tuple:
            nProt = np.array(nProt)
        if type(nElec) == int:
            nElec = np.array(nElec)
        if type(nElec) == float:
            nElec = np.array(nElec)
        if type(nElec) == tuple:
            nElec = np.array(nElec)
        if type(height) == int:
            height = np.array(height)
        if type(height) == float:
            height = np.array(height)
        if type(height) == tuple:
            height = np.array(height)
        if type(times) == int:
            time = np.array(times)
        if type(times) == float:
            time = np.array(times)
        if type(times) == tuple:
            time = np.array(times)


        dimsE = nElec.shape
        dimsH = nHyd.shape
        dimsP = nProt.shape
        dimsZ = height.shape

        if ((dimsE != dimsH) or (dimsH != dimsP) or (dimsP != dimsZ)):
            print('\n>>> YOUR VARIABLES ARE NOT ALL THE SAME DIMENSION:\n')
            print('     dimsE = ',dimsE) 
            print('     dimsH = ',dimsH) 
            print('     dimsP = ',dimsP) 
            print('     dimsZ = ',dimsZ)
            sys.exit('Exiting... enter the correct inputs to correct this issue')

    ########################################################################
    # Assign the output
    ########################################################################
    
        self.nHyd = nHyd
        self.nElec = nElec
        self.nProt = nProt
        self.height = height
        self.times = times
        self.ndims = dimsE
        self.units = 'number densities in [cm^-3]'
        


class SuprathermalParticles:
    """
     
    Holds the properties of the suprathermal protons present
    in the simulation. 
   
    Inputs
    _______

    nthmp_e   -- float
                 Energy grid, in keV
    nthmp_mu  -- float
                 Pitch angle grid, mu (can be ommitted if mu is not resolved)
    nthmp_f   -- float
                 particle distribution, in protons/cm^3/sr/keV
    ionfrac -- float
               The fraction of beam particles that are the ion of 
               interest. For example, if you are looking at He 304 
               emission you will want to know the alpha particle
               flux. One way to do this is to set it as some fraction 
               of the total beam particle flux (alternatively you can 
               input the alpha particle flux directly via nthmp_ft).
               [Optional, default is 1]
    These can be either 
    1) single values (i.e. one depth point at one time),
    2) 1D array of values (i.e. many depth points at one time)
    3) 2D array of values where multiple depth points are defined
       at multiple times. 
    
    Outputs
    _______

    An object containing the particle beam properties, integrated in over
    pitch angle in protons/cm^3/keV.


    Notes
    _____

    If the input is in 1D (i.e. pitch angle grid is not resolved) then nthmp_mu 
    equals zero and the full distribution function is reduced to the energy 
    distribution just by multiplying by 4pi. 

    If the input does have pitch angle resolved then the distribution function 
    is integrated over pitch angle. NOT YET IMPLEMENTED, TO BE DONE, Graham Kerr
    August 2021.


    Graham Kerr
    July 2021

    """

    def __init__(self, nthmp_e = 0, nthmp_mu = 0, nthmp_f = 0, 
                       ionfrac = 1.0):

    ########################################################################
    # Some preliminary set up
    ########################################################################

        ###
        ### Check the dimensions
        ###
        ## Turn to np array if an integer or float are provided
        if type(nthmp_e) == int:
            nthmp_e = np.array(nthmp_e)
        if type(nthmp_e) == float:
            nthmp_e = np.array(nthmp_e)
        if type(nthmp_e) == tuple:
            nthmp_e = np.array(nthmp_e)
        if type(nthmp_mu) == int:
            nthmp_mu = np.array(nthmp_mu)
        if type(nthmp_mu) == float:
            nthmp_mu = np.array(nthmp_mu)
        if type(nthmp_mu) == tuple:
            nthmp_mu = np.array(nthmp_mu)
        if type(nthmp_f) == int:
            nthmp_f = np.array(nthmp_f)
        if type(nthmp_f) == float:
            nthmp_f = np.array(nthmp_f)
        if type(nthmp_f) == tuple:
            nthmp_f = np.array(nthmp_f)

        dimsEn = nthmp_e.shape
        dimsMu = nthmp_mu.shape
        dimsF = nthmp_f.shape
    
        #### Insert some check on dimensions here!
   

    ########################################################################
    # Calculate particles / cm^3 / keV
    ########################################################################

        ## Is pitch angle grid resolved? 
        if len(nthmp_mu) <= 2:
            fe = 4.0*constants.pi*nthmp_f*ionfrac
        elif len(nthmp_mu) > 2:
            fe = nthmp_f * 0.0 ### WILL BE IMPLEMENTED IN THE FUTURE. SET TO ZERO 
                           ### FOR NOW TO AVOID MISTAKES IF A RESOLVED GRID IS
                           ### INPUT


    ########################################################################
    # Assign the output
    ########################################################################

        self.fe = fe
        self.e = nthmp_e
        self.mu = nthmp_mu
        self.ionfrac = ionfrac
        self.units = 'Energy in [keV], distribution fn in [particles/cm^3/keV]'


