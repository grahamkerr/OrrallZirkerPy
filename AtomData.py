import numpy as np 
# import OrrallZirkerPy as OZpy
from OrrallZirkerPy.CrossSections import CrossSecH
from scipy import constants

"""
The 'main' class CSecActive stores the cross sections in arrays of transition 
i -> j for use when solving the equations later.  

It does this from another object that utilises the fits held in CrossSections.py 
that basically collates the chosen cross sections from the various sources, 

It is *meant* to be straightforward to swap in different
cross sections if required, but we will see how that pans out.  
 
Currently there are two options: a 3 level Hydrogen atom (n = 1, 2, 3), and 
a 2 level Hydrogen atom (n = 1, 2). The arrays span [nLev+1, nLev+1] since we 
want to include the continuum also. **SHOULD I JUST CHANGE THIS TO INCLUDE THE
CONTINUUM FROM THE START?**

Graham Kerr
July 2021

"""

class Transitions:

    def __init__(self, nLev = 3, species = 'H'):
        """
        An object to hold atomic data needed to compute the emission:

        wavelegnth_rest -- the rest wavelength of the line
        upplev -- the upper level number of our model atom
        lowlev -- the lower level number of our model atom
        Aji -- the Einstein A coeff. of the transition 

        Inputs
        ______

        nLev = the number of levels; selects either a 3 or 2 level 
               hydrogen atom (default is nLev = 3)

        Notes
        _____


        Graham Kerr
        August 2021

        """

        if species == 'H':

            if nLev == 3:

                wavelength_rest = np.array((1215.6701,1025.728,6562.79))
                upplev = np.array((1, 2, 2))
                lowlev = np.array((0, 0, 1))
                Aji = np.array((4.69800e8,5.57503e7,4.41018e7 ))
                phot2erg = constants.h*1e7 * constants.c*1e10 / wavelength_rest#for c in angstrom/s; wavelength angstroms; h in erg s

                self.wavelength_rest = wavelength_rest
                self.upplev = upplev
                self.lowlev = lowlev
                self.Aji = Aji
                self.phot2erg = phot2erg
                self.mass = constants.value('proton mass energy equivalent in MeV')*1e3 #KeV/c^2
                self.nLev = nLev
                self.species = species

            elif nLev == 2:

                wavelength_rest = np.array((1215.6701))
                upplev = np.array((1))
                lowlev = np.array((0))
                Aji = np.array((4.69800e8))
                phot2erg = constants.h*1e7 * constants.c*1e10 / wavelength_rest#for c in angstrom/s; wavelength angstroms; h in erg s

                self.wavelength_rest = wavelength_rest
                self.upplev = upplev
                self.lowlev = lowlev
                self.Aji = Aji
                self.phot2erg = phot2erg
                self.mass = constants.value('proton mass energy equivalent in MeV')*1e3 #KeV/c^2
                self.nLev = nLev
                self.species = species



class EinsteinA:

    def __init__(self, nLev = 3, species = 'H'):

        """
        An object to store the Einstein A values -- the inverse lifetimes
        of each transition.

        Inputs
        ______

        nlev -- int
                Number of bound levels in the model atom
                Optional, default = 3

        Outputs
        _______

        An object containing the Aij values in /s

        Notes
        _____
        
        To fit the format of the collisional transitions, Aij is [nLev+1, nLev+1]
        in size. However, most entries are zero. 

        Graham Kerr
        July 2021

        """
    ########################################################################
    # Some preliminary set up
    ########################################################################

        ## Some param checks
        if species == 'H':
            if (nLev != 2) and (nLev !=3):
        	    print('\n>>> You have not entered a valid value of nLev (for H, nLev = 2 or 3)\n     Defaulting to nLev = 3!')
        	    nLev = 3
       
        Aij = np.zeros([nLev+1,nLev+1],dtype = np.float64)

    ########################################################################
    # Assign the values
    ########################################################################
    
        if species == 'H':
            Aij[1,0] = 4.69800e8 ## Lyman alpha n = 2 --> n = 1
    
            if nLev == 3:
                Aij[2,0] = 5.57503e7 ## Lyman beta n = 3 --> n = 1
                Aij[2,1] = 4.41018e7 ## Balmer alpha n = 3 --> n = 2

            self.nLev = nLev
            self.Aij = Aij
        

 
class CSecActive:


    def __init__(self, energy, nLev = 3, species = 'H'):
        """

        An object to store the cross sections in arrays [i,j,E], e.g. a 
        transition lev 1 to lev 2 as a function of energy would be 
        [0,1,:].

        Inputs
        ______

        energy -- the energies at which to evaluate the cross sections. 
                  float
                  Can be either a single value or array. 
        nLev   -- the number of bound levels in the atom/ion. 
                  int
                  optional; default nLev = 3

        Outputs
        _______

        An object holding charge exchange (CX) and collisional excitation/ionisation
        (protons: colP, hydrogen atoms: colH, electrons: colE) cross sections, in 
        array format [i,j,E]. Dimensions are [nLev+1, nLev+1, nE].

        NOTES
        _____

        Pulls the cross sections from CSec_ level objects (see below in this file).

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

        ### Assign the cross sections for Hydrogen
        if species == 'H':
            if (nLev != 2) and (nLev !=3):
        	    print('\n>>> You have not entered a valid value of nLev (for H, nLev = 2 or 3)\n     Defaulting to nLev = 3!')
        	    nLev = 3
   

            self.energy = energy
            self.nE = len(self.energy)
            self.nLev = nLev
            self.nLev_plus_cont = nLev+1
            self.species = species
            self.Units = 'energy in [keV], Q in [10^-17 cm^-2]'

            if nLev == 3:
            
                ## Initialise the object
                csecs = CSec_H3lev(energy)
                cs_colH = np.zeros([nLev+1, nLev+1, self.nE], dtype=np.float64)
                cs_colP = np.zeros([nLev+1, nLev+1, self.nE], dtype=np.float64)
                cs_colE = np.zeros([nLev+1, nLev+1, self.nE], dtype=np.float64)
                cs_CX =   np.zeros([nLev+1, nLev+1, self.nE], dtype=np.float64)
 
                ### Assign Charge Ex.
                cs_CX[3,0,:] =  csecs.Q_p1      
                cs_CX[3,1,:] =  csecs.Q_p2  
                cs_CX[3,2,:] =  csecs.Q_p3  

                #### Proton collisionsal excitation/ionisatiom
                cs_colP[0,1,:] =  csecs.Q_12P
                cs_colP[0,2,:] =  csecs.Q_13P
                cs_colP[0,3,:] =  csecs.Q_1pP
                cs_colP[1,2,:] =  csecs.Q_23P
                cs_colP[1,3,:] =  csecs.Q_2pP
                cs_colP[2,3,:] =  csecs.Q_3pP

                #### Electron collisionsal excitation/ionisatiom
                cs_colE[0,1,:] =  csecs.Q_12E
                cs_colE[0,2,:] =  csecs.Q_13E
                cs_colE[0,3,:] =  csecs.Q_1pE
                cs_colE[1,2,:] =  csecs.Q_23E
                cs_colE[1,3,:] =  csecs.Q_2pE
                cs_colE[2,3,:] =  csecs.Q_3pE

                #### Hydrogen collisionsal excitation/ionisatiom
                cs_colH[0,1,:] =  csecs.Q_12H 
                cs_colH[0,2,:] =  csecs.Q_13H
                cs_colH[0,3,:] =  csecs.Q_1pH
                # cs_colH[1,2,:] =  csecs.Q_23H 
                # cs_colH[1,3,:] =  csecs.Q_2pH 
                # cs_colH[2,3,:] =  csecs.Q_3pH 

            elif nLev == 2:

        	    ## Initialise the object
                csecs = CSec_H2lev(energy)
                cs_colH = np.zeros([nLev+1, nLev+1, self.nE], dtype=np.float64)
                cs_colP = np.zeros([nLev+1, nLev+1, self.nE], dtype=np.float64)
                cs_colE = np.zeros([nLev+1, nLev+1, self.nE], dtype=np.float64)
                cs_CX =   np.zeros([nLev+1, nLev+1, self.nE], dtype=np.float64)
 
                ### Assign Charge Ex.
                cs_CX[2,0,:] =  csecs.Q_p1      
                cs_CX[2,1,:] =  csecs.Q_p2  

                #### Proton collisionsal excitation/ionisatiom
                cs_colP[0,1,:] =  csecs.Q_12P
                cs_colP[0,2,:] =  csecs.Q_1pP
                cs_colP[1,2,:] =  csecs.Q_2pP

                #### Electron collisionsal excitation/ionisatiom
                cs_colE[0,1,:] =  csecs.Q_12E
                cs_colE[0,2,:] =  csecs.Q_12E
                cs_colE[1,2,:] =  csecs.Q_2pE

                #### Hydrogen collisionsal excitation/ionisatiom
                cs_colH[0,1,:] =  csecs.Q_12H 
                cs_colH[0,2,:] =  csecs.Q_1pH
                # cs_colH[1,2,:] =  csecs.Q_2pH 

                   
            self.cs_CX = cs_CX
            self.cs_colP = cs_colP
            self.cs_colE = cs_colE
            self.cs_colH = cs_colH


class CSec_H3lev:
    """
     
    Hydrogen 3 level atom cross sections (n = 1, 2, 3) for 
    Orrall Zirker Effcet calculations. 
   
    Inputs
    _______

    energy -- projectile energy in keV (single value or array)

    Outputs
    _______

    An object with energy in keV and cross sections in 10^-17 cm^2,
    for the cross sections detailed in the comments below.

    Notes
    _____

    This class holds the cross sections to be used in the calculation 
    of the Orrall-Zirker effect. 

    It utilises the fits held in CrossSections.py and returns a similar 
    structure, but this is designed to make it easy to swap in different
    cross sections if required, without having to dig through lots of code. 

    A projectile energy is input in keV.

    Graham Kerr
    July 2021

    """
	
    def __init__(self, energy):
    
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

        self.energy = energy
        self.nE = len(self.energy)
        self.nlev = 3
        self.Units = 'energy in [keV], Q in [10^-17 cm^-2]'



    ########################################################################
    # Grab some data from CrossSections.py
    ########################################################################
        
        ## Create the CrossSections object
        cs = CrossSecH(self.energy)

        ## Most of the cross sections I want are held in this the kerr_poly
        ## class
        kerr_poly = cs.cs_kerr_poly()


    ########################################################################
    # Assign the cross sections
    ########################################################################

        ####################
        ### CHARGE EXCHANGE
        ####################
        Q_p1 = kerr_poly.Q_p1
        Q_p2 = kerr_poly.Q_p2
        Q_p3 = kerr_poly.Q_p3


        #####################################
        ### COLLISIONAL EXCITATION/IONISATION
        #####################################

        ### 
        ### n = 1 -> proton
        ###
        Q_1pP = kerr_poly.Q_1pP
        Q_1pH = kerr_poly.Q_1pH
        Q_1pE = kerr_poly.Q_1pE

        ###
        ### n = 1 -> n = 2
        ###
        Q_12P = kerr_poly.Q_12P
        Q_12H = kerr_poly.Q_12H
        Q_12E = kerr_poly.Q_12E

        ###
        ### n = 1 -> n = 3
        ###
        Q_13P = kerr_poly.Q_13P
        Q_13H = kerr_poly.Q_13H
        Q_13E = kerr_poly.Q_13E

        ###
        ### n = 2 -> proton
        ###
        Q_2pP = kerr_poly.Q_2pP
        # Q_2pH = kerr_poly.Q_2pH
        Q_2pE = kerr_poly.Q_2pE

        ###
        ### n = 2 -> n = 3
        ###
        Q_23P = kerr_poly.Q_23P
        # Q_23H = kerr_poly.Q_23H
        Q_23E = kerr_poly.Q_23E

        ###
        ### n = 3 -> proton
        ###
        Q_3pP = kerr_poly.Q_3pP
        # Q_3pH = kerr_poly.Q_3pH
        Q_3pE = kerr_poly.Q_3pE

    ########################################################################
    # Define the output
    ########################################################################

        self.Q_p1 = Q_p1
        self.Q_p2 = Q_p2
        self.Q_p3 = Q_p3
      
        self.Q_1pP = Q_1pP
        self.Q_1pH = Q_1pH
        self.Q_1pE = Q_1pE

        self.Q_12P = Q_12P
        self.Q_12H = Q_12H
        self.Q_12E = Q_12E

        self.Q_13P = Q_13P
        self.Q_13H = Q_13H
        self.Q_13E = Q_13E

        self.Q_2pP = Q_2pP
        # self.Q_2pH = Q_2pH
        self.Q_2pE = Q_2pE

        self.Q_23P = Q_23P
        # self.Q_23H = Q_23H
        self.Q_23E = Q_23E

        self.Q_3pP = Q_3pP
        # self.Q_3pH = Q_3pH
        self.Q_3pE = Q_3pE


class CSec_H2lev:
    """
     
    Hydrogen 2 level atom cross sections (n = 1, 2) for 
    Orrall Zirker Effcet calculations. 
   

    Inputs
    _______

    energy -- projectile energy in keV (single value or array)

    Outputs
    _______

    An object with energy in keV and cross sections in 10^-17 cm^2,
    for the cross sections detailed in the comments below.

    Notes
    ______

    This class holds the cross sections to be used in the calculation 
    of the Orrall-Zirker effect. 

    It utilises the fits held in CrossSections.py and returns a similar 
    structure, but this is designed to make it easy to swap in different
    cross sections if required, without having to dig through lots of code. 

    A projectile energy is input in keV.

    Graham Kerr
    July 2021

    """
	
    def __init__(self, energy):
    
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

        self.energy = energy
        self.nE = len(self.energy)
        self.nlev = 2
        self.Units = 'energy in [keV], Q in [10^-17 cm^-2]'



    ########################################################################
    # Grab some data from CrossSections.py
    ########################################################################
        
        ## Create the CrossSections object
        cs = CrossSecH(self.energy)

        ## Most of the cross sections I want are held in this the kerr_poly
        ## class
        kerr_poly = cs.cs_kerr_poly()


    ########################################################################
    # Assign the cross sections
    ########################################################################

        ####################
        ### CHARGE EXCHANGE
        ####################
        Q_p1 = kerr_poly.Q_p1
        Q_p2 = kerr_poly.Q_p2


        #####################################
        ### COLLISIONAL EXCITATION/IONISATION
        #####################################

        ### 
        ### n = 1 -> proton
        ###
        Q_1pP = kerr_poly.Q_1pP
        Q_1pH = kerr_poly.Q_1pH
        Q_1pE = kerr_poly.Q_1pE

        ###
        ### n = 1 -> n = 2
        ###
        Q_12P = kerr_poly.Q_12P
        Q_12H = kerr_poly.Q_12H
        Q_12E = kerr_poly.Q_12E

        ###
        ### n = 2 -> proton
        ###
        Q_2pP = kerr_poly.Q_2pP
        # Q_2pH = kerr_poly.Q_2pH
        Q_2pE = kerr_poly.Q_2pE

        ###
        ### n = 2 -> n = 3
        ###
        Q_23P = kerr_poly.Q_23P
        # Q_23H = kerr_poly.Q_23H
        Q_23E = kerr_poly.Q_23E




    ########################################################################
    # Define the output
    ########################################################################

        self.Q_p1 = Q_p1
        self.Q_p2 = Q_p2
      
        self.Q_1pP = Q_1pP
        self.Q_1pH = Q_1pH
        self.Q_1pE = Q_1pE

        self.Q_12P = Q_12P
        self.Q_12H = Q_12H
        self.Q_12E = Q_12E

        self.Q_2pP = Q_2pP
        # self.Q_2pH = Q_2pH
        self.Q_2pE = Q_2pE

        self.Q_23P = Q_23P
        # self.Q_23H = Q_23H
        self.Q_23E = Q_23E


