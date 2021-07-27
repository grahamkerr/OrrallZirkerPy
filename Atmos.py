import numpy as np

class ambientpops:
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
    time  -- Simulation time in seconds (cam be omitted)

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

        if ((dimsE == dimsH) and (dimsH == dimsP) and (dimsP == dimsZ)):
            equal = True
        else:
            equal = False
            print('\n>>> YOUR VARIABLES ARE NOT ALL THE SAME DIMENSION:\n')
            print('     dimsE = ',dimsE) 
            print('     dimsH = ',dimsH) 
            print('     dimsP = ',dimsP) 
            print('     dimsZ = ',dimsZ)
            print('     Returning zeros...\n') 

    ########################################################################
    # Assign the output
    ########################################################################
    
        if equal == True:
            self.nHyd = nHyd
            self.nElec = nElec
            self.nProt = nProt
            self.height = height
            self.time = times
            self.ndims = dimsE
            self.units = 'number densities in [cm^-3]'
        elif equal == False:
            self.nHyd = 0
            self.nElec = 0
            self.nProt = 0
            self.height = 0
            self.time = 0
            self.ndims = 0
            self.units = 'number densities in [cm^-3]'