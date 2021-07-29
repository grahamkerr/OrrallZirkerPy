import numpy as np 
import OrrallZirkerPy as OZpy

## These classes hold the cross sections to be used in the calculation 
## of the Orrall-Zirker effect. 

## They utilise the fits held in CrossSections.py and returns a similar 
## structure, but this script is designed to make it easy to swap in different
## cross sections if required, without having to dig through lots of code. 

## There is one object for a 3 level Hydrogen atom, and one for a 2 level 
## Hydrogen atom. 

class CrossSec_H3lev:
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
        cs = OZpy.CrossSections.CrossSec(self.energy)

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


class CrossSec_H2lev:
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
        cs = OZpy.CrossSections.CrossSec(self.energy)

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


