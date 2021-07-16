import numpy as np 
from numpy.polynomial.polynomial import Polynomial as Poly
from numpy.polynomial.chebyshev import Chebyshev as Cheb
from scipy.optimize import curve_fit

class CrossSec:

    '''
    This class takes a proton energy, either a single value or a list, 
    and has various methods to compute cross sections of charge exchange 
    interactions between that energetic proton with ambient hydrogen, and 
    relevant processes that determine the interaction of the resulting 
    energetic neutral hydrogen with the ambient plasma.

    Each cross section has different valid energy ranges, and have been extended 
    via fitting to go to arbrotarily high energy (though of course those extrapolations 
    should be used with care). These cross sections should not be used for interactions with 
    protons on order of eV energy. 


    '''

    def __init__(self, energy):
        
        ## Turn to np array if an integer or float are provided
        if type(energy) == int:
            energy = np.array(energy)
        if type(energy) == float:
            energy = np.array(energy)
        if type(energy) == tuple:
            energy = np.array(energy)

        self.energy = energy
        self.nE = len(self.energy)



    def cs_bw99(self):

        '''
        This function will calculate the cross-sections required 
        to compute the population of suprathermal neutral hydrogen,
        given an energy E in keV. 

        Those cross sections are: 

        Q_pj -- the charge exchange cs from protons to H level j = 1,2

        Q_ijP, Q_ijH, Q_oijE -- the collisional ionisation or excitation 
                                cross sections from i-->j 

        The following cross sections are computed:

        Q_p1 -> charge exchange to n=1 (ground)
        Q_p2 -> charge exchange to n=2 
        Q_1pP -> collisional ionisation via protons n=1 to proton
        Q_1pH -> collisional ionisation via hydrogen n=1 to proton
        Q_1pE -> collisional ionisation via electrons n=1 to proton
        Q_12P -> collisional excitation via protons n=1 to n=2
        Q_12H -> collisional excitation via hydrogen n=1 to n=2
        Q_12E -> collisional excitation via electrons n=1 to n=2
        Q_23E -> collisional excitation via electrons n=2 to n=3

        For details, including references to the data from which the polynomials 
        were derived, see Brosius & Woodgate 1999, ApJ 514

        Parameters
        __________

        energy : float
            The energy at which to compute the cross sections, in keV


        Outputs
        _________

        cross_secs : 
            An object containing each cross section, in 10^-17 cm^2 


        Notes
        ________

        BW99 only computed the cross-sections required for Lyman alpha. 
        For the Lyman beta or H-alpha problem we need additional rates.

        Graham Kerr
        July 2021

        '''

        ########################################################################
        # Some preliminary set up
        ########################################################################

        ## Turn to list if an integer or float are provided
        # if type(self.energy) == int:
        #     self.energy = [self.energy]
        # if type(self.energy) == float:
        #     self.energy = [self.energy]

        ## How many energies to calculate
        # nE = len(self.energy)
        
        Q_p1 = np.zeros([self.nE],dtype = np.float64)
        Q_p2 = np.zeros([self.nE],dtype = np.float64)
        Q_1pP = np.zeros([self.nE],dtype = np.float64)
        Q_1pH = np.zeros([self.nE],dtype = np.float64)
        Q_1pE = np.zeros([self.nE],dtype = np.float64)
        Q_12P = np.zeros([self.nE],dtype = np.float64)
        Q_12H = np.zeros([self.nE],dtype = np.float64)
        Q_12E = np.zeros([self.nE],dtype = np.float64)
        Q_23E = np.zeros([self.nE],dtype = np.float64)



        ########################################################################
        # Go through each energy and calculate the cross section
        ########################################################################

        for ind in range(self.nE):

            ########
            # Q_p1
            ########
            Q_p1[ind] = 10.00**(-13.69 - 2.03*np.log10(self.energy[ind]) +
                                         1.39*np.log10(self.energy[ind])**2.0 -
                                         0.827*np.log10(self.energy[ind])**3.0 +
                                         0.0988*np.log10(self.energy[ind])**4.0 
                                )

            ########
            # Q_p2
            ########          
            Q_p2[ind] = 10.00**(-19.02 + 5.59*np.log10(self.energy[ind]) -
                                         2.70*np.log10(self.energy[ind])**2.0 -
                                         0.00586*np.log10(self.energy[ind])**3.0 +
                                         0.0400*np.log10(self.energy[ind])**4.0 
                                )

    
        class cs_bw99_out:
            def __init__(selfout):
                selfout.Q_p1 = Q_p1/1e-17
                selfout.Q_p2 = Q_p2/1e-17
                selfout.energy = self.energy
                selfout.Units = 'energy in [keV], Q in [10^-17 cm^-2]'

        out = cs_bw99_out()

        return out

    def cs_fang95(self):

        '''
        This function will calculate the cross-sections required 
        to compute the population of suprathermal neutral hydrogen,
        given an energy E in keV. 

        Those cross sections are: 

        Q_pj -- the charge exchange cs from protons to H level j = 1,2,3
        Q_ijP, Q_ijH, Q_oijE -- the collisional ionisation or excitation 
                                cross sections from i-->j 

        The following cross sections are computed:

        Q_p1 -> charge exchange to n=1 (ground)
        Q_p2 -> charge exchange to n=2 
        Q_p3 -> charge exchange to n=3
        Q_1pP -> collisional ionisation via protons n=1 to proton
        Q_1pH -> collisional ionisation via hydrogen n=1 to proton
        Q_1pE -> collisional ionisation via electrons n=1 to proton
        Q_12P -> collisional excitation via protons n=1 to n=2
        Q_12H -> collisional excitation via hydrogen n=1 to n=2
        Q_12E -> collisional excitation via electrons n=1 to n=2
        Q_13P -> collisional excitation via protons n=1 to n=3
        Q_13H -> collisional excitation via hydrogen n=1 to n=3
        Q_13E -> collisional excitation via electrons n=1 to n=3
        Q_23E -> collisional excitation via electrons n=2 to n=3

        For details, including references to the data from which the polynomials 
        were derived, see Fang, Feautrier & Henoux et al 1995 A&A 297, 854

        Parameters
        __________

        energy : float
            The energy at which to compute the cross sections, in keV


        Outputs
        _________

        cross_secs : 
             An object containing each cross section, in 10^-17 cm^2 


        Graham Kerr
        July 2021

        '''

        ########################################################################
        # Some preliminary set up
        ########################################################################
 
        ## Turn to list if an integer or float are provided
        # if type(self.energy) == int:
        #     self.energy = [self.energy]
        # if type(self.energy) == float:
        #     self.energy = [self.energy]

        ## How many energies to calculate
        # nE = len(self.energy)
        
        Q_p1 = np.zeros([self.nE],dtype = np.float64)
        Q_p2 = np.zeros([self.nE],dtype = np.float64)
        Q_p3 = np.zeros([self.nE],dtype = np.float64)
        Q_1pP = np.zeros([self.nE],dtype = np.float64)
        Q_1pH = np.zeros([self.nE],dtype = np.float64)
        Q_1pE = np.zeros([self.nE],dtype = np.float64)
        Q_12P = np.zeros([self.nE],dtype = np.float64)
        Q_12H = np.zeros([self.nE],dtype = np.float64)
        Q_12E = np.zeros([self.nE],dtype = np.float64)
        Q_13P = np.zeros([self.nE],dtype = np.float64)
        Q_13H = np.zeros([self.nE],dtype = np.float64)
        Q_13E = np.zeros([self.nE],dtype = np.float64)
        Q_23E = np.zeros([self.nE],dtype = np.float64)

        ########################################################################
        # Go through each energy and calculate the cross section
        ########################################################################

        for ind in range(self.nE):

            # print(self.energy[ind])

            ########
            # Q_p1
            ########
            if self.energy[ind] <= 16.0:
                a0,a1,a2,a3 = 192.53,-30.481,2.8448,-9.4366e-2 
                Q_p1[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3)
            elif self.energy[ind] > 16.0:
                a0,a1 = 82.420, -0.03564
                Q_p1[ind] = a0*np.exp(a1*self.energy[ind])

            ########
            # Q_p2
            ########
            if self.energy[ind] <= 18.0:
                a0,a1,a2,a3 = 2.4742, -0.018978, 4.6428e-2, -1.8790e-3
                Q_p2[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3)
            elif self.energy[ind] > 18.0:
                a0,a1,a2,a3 = 10.016, -0.23804, 1.7660e-3, -4.1271e-6
                Q_p2[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3)


            ########
            # Q_p3
            ########
            if self.energy[ind] <= 16.0:
                a0,a1,a2,a3,a4 = 1.1452, -1.33788, 3.9887e-1, -3.6992e-2, 1.0810e-3
                Q_p3[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3) + (a4*self.energy[ind]**4)
            elif self.energy[ind] > 16.0:
                a0,a1,a2,a3 = 2.0134, -0.03773, 2.2500e-4, -4.3204e-7
                Q_p3[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3)

            ########
            # Q_1pP
            ########
            if self.energy[ind] <= 80.0:
                a0,a1,a2,a3 = -6.9860, 0.95306, -1.2987e-2, 5.0484e-5
                Q_1pP[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3) 
            elif self.energy[ind] > 80.0:
                a0,a1,a2,a3 = 16.472, -0.06162, 9.2635e-5, -4.5990e-8
                Q_1pP[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3)

            ########
            # Q_1pH
            ########
            if self.energy[ind] <= 15.0:
                a0,a1,a2,a3 = 0.29922, 0.19057, 1.5668e-1, -8.4700e-3
                Q_1pH[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3) 
            elif self.energy[ind] > 15.0:
                a0,a1,a2,a3 = 11.962, -0.13523, -1.5500e-4, 5.9890e-6
                Q_1pH[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3)

            ########
            # Q_1pE
            ########
            if self.energy[ind]/1e3 <= 63.0:
                a0,a1,a2,a3,a4 = -14.326, 1.6028, -4.9415e-2, 7.0030e-4, -3.7820e-6
                Q_1pE[ind] = a0 + (a1*(self.energy[ind]/1e3)**1) + (a2*(self.energy[ind]/1e3)**2) + (a3*(self.energy[ind]/1e3)**3) + (a4*(self.energy[ind]/1e3)**4)
            elif self.energy[ind]/1e3 > 63.0:
                a0,a1,a2,a3 = 7.4762, -0.02284, 3.0692e-5, -1.4225e-8
                Q_1pE[ind] = a0 + (a1*(self.energy[ind]/1e3)**1) + (a2*(self.energy[ind]/1e3)**2) + (a3*(self.energy[ind]/1e3)**3)

            ########
            # Q_12P
            ########
            if self.energy[ind] <= 16.0:
                a0,a1,a2,a3 = 1.3956, 1.0669, -1.4844e-1, 5.6340e-3
                Q_12P[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3) 
            elif self.energy[ind] > 16.0:
                a0,a1,a2,a3 = 0.42435, 0.27398, -2.4905e-3, 6.3147e-6
                Q_12P[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3)

            ########
            # Q_12H
            ########
            if self.energy[ind] <= 15.0:
                a0,a1,a2,a3 = 1.9743, 0.73240, -8.4920e-2, 3.0500e-3 
                Q_12H[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3) 
            elif self.energy[ind] > 15.0:
                a0,a1,a2,a3 = 7.6956, -0.27701, 3.4650e-3, -1.4478e-5
                Q_12H[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3)

            ########
            # Q_12E
            ########
            a0,a1,a2,a3 = 1.9988, 0.26822, -4.2150e-3, 1.9194e-5
            Q_12E[ind] = a0 + (a1*(self.energy[ind]/1e3)**1) + (a2*(self.energy[ind]/1e3)**2) + (a3*(self.energy[ind]/1e3)**3) 

            ########
            # Q_13P
            ########
            if self.energy[ind] <= 16.0:
                a0,a1,a2,a3 = 5.58e-4, 0.037722, -3.4860e-3, 2.2200e-4
                Q_13P[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3) 
            elif self.energy[ind] > 16.0:
                a0,a1,a2,a3,a4 = -2.3810, 0.25882, -5.0270e-3, 4.0225e-5, -1.3794e-7
                Q_13P[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3) + (a4*self.energy[ind]**4)
            elif self.energy[ind] > 100.0:
                a0,a1,a2,a3,a4,a5 = -2.3810, 0.25882, -5.0270e-3, 4.0225e-5, -1.3794e-7, 1.6189e-10
                Q_13P[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3) + (a4*self.energy[ind]**4) + (a5*self.energy[ind]**5)

            ########
            # Q_13H
            ########
            if self.energy[ind] <= 15.0:
                a0,a1,a2,a3 = 0.23215, 0.34071, -2.0600e-2, 1.7400e-4
                Q_13H[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3) 
            elif self.energy[ind] > 15.0:
                a0,a1,a2,a3 = 2.3376, -0.07657, 8.6900e-4, -3.2869e-6
                Q_13H[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3)

            ########
            # Q_13E
            ########
            if self.energy[ind]/1e3 <= 30.0:
                a0,a1,a2,a3 = -7.3427, 1.2622, -5.6240e-2, 7.8900e-4
                Q_13E[ind] = a0 + (a1*(self.energy[ind]/1e3)**1) + (a2*(self.energy[ind]/1e3)**2) + (a3*(self.energy[ind]/1e3)**3) 
            elif self.energy[ind]/1e3 > 30.0:
                a0,a1,a2,a3 = 0.89060, 0.002260, -3.4400e-4, 1.4179e-6
                Q_13E[ind] = a0 + (a1*(self.energy[ind]/1e3)**1) + (a2*(self.energy[ind]/1e3)**2) + (a3*(self.energy[ind]/1e3)**3)

            ########
            # Q_23E
            ########
            a0,a1,a2,a3 =  267.13, -3.2192, 1.2188e-2, 6.3314e-6
            Q_23E[ind] = a0 + (a1*(self.energy[ind]/1e3)**1) + (a2*(self.energy[ind]/1e3)**2) + (a3*(self.energy[ind]/1e3)**3) 
            
        
        ## A class to output the cross sections
        class cs_fang95_out:
            def __init__(selfout):
                selfout.Q_p1 = Q_p1
                selfout.Q_p2 = Q_p2
                selfout.Q_p3 = Q_p3
                selfout.Q_1pP = Q_1pP
                selfout.Q_1pH = Q_1pH
                selfout.Q_1pE = Q_1pE
                selfout.Q_12P = Q_12P
                selfout.Q_12H = Q_12H
                selfout.Q_12E = Q_12E
                selfout.Q_13P = Q_13P
                selfout.Q_13H = Q_13H
                selfout.Q_13E = Q_13E
                selfout.Q_23E = Q_23E
                selfout.energy = self.energy
                selfout.Units = 'energy in [keV], Q in [10^-17 cm^-2]'

        out = cs_fang95_out()

        return out


class cs_cheshire70:

    '''
    This class holds the energy and cross sections for charge transfer 
    from Cheshire et al 1970 J. Phys. B, 3 813, Table 5.
   
    Cross sections are of collisions between protons and neutral H to 

    1s : Q_p1
    2s : Q_p_2s
    2p : Q_p_2p
    2s+2p: Q_p2
 
    Energy range 1-1000 keV

    '''
        
    def __init__(self):
        self.energy = np.array((1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 
                                20.0, 25.0, 30.0, 40.0, 60.0, 
                                70.0, 100.0, 300.0, 1000.0))
        self.Q_p1 = np.array((1.912e2, 1.476e2, 1.131e2, 8.989e1, 7.775e1, 5.819e1, 
                     4.140e1, 2.927e1, 2.090e1, 1.129e1, 4.303e0, 2.738e0, 
                     8.889e-1, 8.507e-3, 2.628e-5))
        self.Q_p_2s = np.array((1.153e-1, 3.466e-1, 4.133e-1, 8.024e-1, 1.940e0, 
                       3.089e0, 3.760e0, 3.726e0, 3.345e0, 2.488e0, 1.184e0, 
                       7.984e-1, 2.634e-1, 1.919e-3, 4.120e-6))
        self.Q_p_2p = np.array((2.224e0, 2.683e0, 2.983e0, 3.189e0, 3.309e0,
                       2.029e0, 1.633e0, 1.590e0, 1.417e0, 9.142e-1, 
                       3.844e-1, 2.424e-1, 5.578e-2, 1.964e-4, 1.466e-7))
        self.Q_p2 = self.Q_p_2s + self.Q_p_2p
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'


class cs_ludde82:

    '''
    This class holds the energy and cross sections for charge transfer 
    from Ludde et al 1982 J. Phys. B, 15 2703, Table 1.
   
    Cross sections are of collisions between protons and neutral H to 

    1s : Q_p1
    2s : Q_p_2s
    2p : Q_p_2p
    2s+2p: Q_p2
    n=3 : Q_p3
 
    Energy range 1-50 keV

    '''
        
    def __init__(self):
        self.energy = np.array((1.0,2.0,4.0,6.0,8.0,14.0,16.0,25.0,50.0))
        self.Q_p1 = np.array((147.00, 123.00, 103.00, 85.0, 79.20, 59.0, 53.70, 31.30, 3.50))
        self.Q_p_2s = np.array((0.04, 0.13, 0.34, 0.59, 1.20, 2.15, 2.40, 2.77, 0.87))
        self.Q_p_2p = np.array((2.20, 2.86, 2.67, 2.85, 3.15, 4.05, 3.86, 1.26, 0.53))
        self.Q_p2 = np.array((2.24, 2.99, 3.01, 3.44, 4.35, 6.20, 6.26, 4.03, 1.40))
        self.Q_p3 = np.array((0.02, 0.03, 0.13, 0.50, 1.75, 0.54, 1.21, 1.77, 0.18))
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'


class cs_shakeshaft78:

    '''
    This class holds the energy and cross sections for charge transfer 
    from Shakeshaft 1978 Phys. Rev. A, 18, Table 2
   
    Cross sections are of collisions between protons and neutral H to 

    1s : Q_p1
    2s : Q_p_2s
    2p0 : Q_p_2p0
    2p1 : Q_p_2p1
    n=2: sum of 2s and 2p states, Q_p2
    3s : Q_p_3s
    3p0 : Q_p_3p0
    3p1 : Q_p_3p1
    3d0 : Q_p_3d0
    3d1 : Q_p_3d1
    3d2 : Q_p_3d2
    n=3 : sum of 3s, 3p, 3d states, Q_p3
 
    Energy range 15-200 keV

    '''
        
    def __init__(self):
        self.energy = np.array((15.0, 25.0, 40.0, 50.0, 60.0, 75.0, 145.0, 200.0))
        self.Q_p1 = np.array((58.35, 30.35, 11.89, 6.78, 4.10, 2.10, 0.19, 0.047))
        self.Q_p_2s = np.array((3.41, 3.98, 2.33, 1.39, 0.82, 0.42, 0.040, 0.0087))
        self.Q_p_2p0 = np.array((0.94, 0.76, 0.39, 0.23, 0.12, 0.049, 0.0054, 0.0010))
        self.Q_p_2p1 = np.array((2.16, 0.98, 0.34, 0.17, 0.089, 0.037, 0.0023, 0.0004))
        self.Q_p2 = self.Q_p_2s + self.Q_p_2p0 + self.Q_p_2p1
        self.Q_p_3s = np.array((0.53, 0.93, 0.67, 0.45, 0.29, 0.14, 0.012, 0.0030))
        self.Q_p_3p0 = np.array((0.33, 0.28, 0.13, 0.077, 0.058, 0.018, 0.0017, 0.0004))
        self.Q_p_3p1 = np.array((0.41, 0.24, 0.11, 0.058, 0.030, 0.012, 0.0008, 0.0001))
        self.Q_p_3d0 = np.array((0.050, 0.031, 0.015, 0.017, 0.017, 0.026, 0.0010, 0.00004))
        self.Q_p_3d1 = np.array((0.14, 0.025, 0.0091, 0.0057, 0.0030, 0.0010, 0.0001, 0.00002))
        self.Q_p_3d2 = np.array((0.0089, 0.0045, 0.0015, 0.0008, 0.0004, 0.0002, 0.00002, 0.000002))
        self.Q_p3 = self.Q_p_3s + self.Q_p_3p0 + self.Q_p_3p1 + self.Q_p_3d0 + self.Q_p_3d1 + self.Q_p_3d2
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

    

class cs_bates53:

    '''
    This class holds the energy and cross sections for charge transfer 
    from Bates & Dalgarno 1953 Proc. Phys. Soc. 66, Table 2
   
    Cross sections are of collisions between protons and neutral H to 

    1s : Q_p1
    2s : Q_p_2s
    2p : Q_p_2p
    n=2: sum of 2s and 2p states, Q_p2
    3s : Q_p_3s
    3p : Q_p_3p
    3d : Q_p_3d
    n=3 : sum of 3s, 3p, 3d states, Q_p3
 
    Energy range  2.5 -- 7900 keV

    '''

    def __init__(self):
        a0 = 5.3e-9
        self.energy = 10.0**np.arange(-1.0, 2.6, 0.1) * 24.97 # keV
        self.Q_p1 = 10.0**np.array((1.73, 1.62, 1.51, 1.39, 1.27, 
                             1.14, 1.00, 0.86, 0.71, 0.54, 
                             0.36, 0.17, -0.05, -0.28, -0.54, 
                             -0.82, -1.13, -1.46, -1.82, -2.21, 
                             -2.62, -3.06, -3.52, -3.99, -4.49, 
                             -5.00, -5.52, -6.06, -6.60, -7.15, 
                             -7.71, -8.28, -8.85, -9.42, -10.00, 
                             -10.58))*np.pi*a0**2*1e17
        self.Q_p_2s = 10**np.array((-0.88, -0.69, -0.54, -0.42, -0.34,
                                    -0.29, -0.28, -0.30, -0.35, -0.43, 
                                    -0.54, -0.68, -0.86, -1.06, -1.30,
                                    -1.57, -1.88, -2.21, -2.59, -2.99,
                                    -3.41, -3.86, -4.33, -4.83, -5.33,
                                    -5.85, -6.48, -6.93, -7.48, -8.03,
                                    -8.60, -9.17, -9.74, -10.32, -10.90,
                                    -11.48))*np.pi*a0**2*1e17
        self.Q_p_2p = 10**np.array((-1.12, -0.84, -0.60, -0.40, -0.24, 
                                    -0.13, -0.07, -0.05, -0.08, -0.15, 
                                    -0.18, -0.45, -0.67, -0.93, -1.24,
                                    -1.59, -1.98, -2.41, -2.88, -3.37,
                                    -3.90, -4.44, -5.01, -5.60, -6.21,
                                    -6.83, -7.46, -8.10, -8.75, -9.41, 
                                    -10.08,-10.74, -11.42, -12.10, -12.78, 
                                    -13.46))*np.pi*a0**2*1e17
        self.Q_p2 = self.Q_p_2s + self.Q_p_2p
        self.Q_p_3s = 10**np.array((-1.86, -1.62, -1.42, -1.25, -1.11, 
                                    -1.02, -0.96, -0.94, -0.96, -1.01,
                                    -1.10, -1.22, -1.38, -1.57, -1.80, 
                                    -2.07, -2.38, -2.72, -3.09, -3.49, 
                                    -3.92, -4.37, -4.85, -5.34, -5.85, 
                                    -6.37, -6.90, -7.45, -8.00, -8.56, 
                                    -9.12, -9.69, -10.27, -10.85, -11.43, 
                                    -12.01))*np.pi*a0**2*1e17
        
        self.Q_p_3p = 10**np.array((-2.18, -1.85, -1.55, -1.30, -1.08,
                                    -0.91, -0.79, -0.72, -0.70, -0.74,
                                    -0.82, -0.97, -1.16, -1.30, -1.70,
                                    -2.04, -2.43, -2.85, -3.31, -3.81, 
                                    -4.33, -4.88, -5.46, -6.05, -6.66,
                                    -7.28, -7.91, -8.55, -9.20, -9.86, 
                                    -10.53, -11.20, -11.87, -12.55, -13.23,
                                    -13.91))*np.pi*a0**2*1e17
        self.Q_p_3d = 10**np.array((-3.43, -3.02, -2.65, -2.31, -2.03, 
                                    -1.80, -1.63, -1.52, -1.48, -1.50, 
                                    -1.58, -1.74, -1.96, -2.25, -2.59,
                                    -2.99, -3.44, -3.94, -4.49, -5.07,
                                    -5.68, -6.32, -6.98, -7.67, -8.37, 
                                    -9.09, -9.82, -10.56, -11.31, -12.06, 
                                    -12.83, -13.60, -14.37, -15.15, -15.92,
                                    -16.71))*np.pi*a0**2*1e17
        self.Q_p3 = self.Q_p_3s + self.Q_p_3p + self.Q_p_3d
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'


class cs_winter09:

    '''
    This class holds the energy and cross sections for charge transfer 
    from Winter 2009 Phys. Rev. A. 80, Table 5. Essentially an update 
    to Shakeshaft 1978.
   
    Cross sections are of collisions between protons and neutral H to 

    1s : Q_p1
    2s : Q_p_2s
    2p : Q_p_2p
    n=2: sum of 2s and 2p states, Q_p2
    3s : Q_p_3s
    3p : Q_p_3p
    3d : Q_p_3d
    n=3 : sum of 3s, 3p, 3d states, Q_p3
 
    Energy range  1 -- 100 keV

    '''

    def __init__(self):
        
        self.energy = np.array((1.0, 1.563, 3.0, 4.0, 5.16, 8.0, 15.0, 25.0, 50.0, 75.0, 100.0))
        self.Q_p1 = np.array((170.7, 153.1, 127.6, 114.5, 103.40, 87.64, 58.35, 30.48, 6.94, 2.10, 0.78))
        self.Q_p_2s = np.array((0.048, 0.080, 0.278, 0.445, 0.676, 1.53, 3.39, 3.97, 1.42, 0.44, 0.15))
        self.Q_p_2p = np.array((2.262, 2.616, 2.861, 2.647, 2.677, 3.17, 3.09, 1.73, 0.37, 0.09, 0.03))
        self.Q_p2 = self.Q_p_2s + self.Q_p_2p
        self.Q_p_3s = np.array((0.002, 0.004, 0.005, 0.007, 0.015, 0.08, 0.52, 0.98, 0.46, 0.15, 0.05))
        self.Q_p_3p = np.array((0.026, 0.047, 0.123, 0.141, 0.180, 0.30, 0.63, 0.54, 0.13, 0.03, 0.01))
        self.Q_p_3d = np.array((0.020, 0.062, 0.120, 0.206, 0.354, 0.30, 0.17, 0.07, 0.01, 0.002, 0.001))
        self.Q_p3 = self.Q_p_3s + self.Q_p_3p + self.Q_p_3d
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

class cs_belkic92:

    '''
    This class holds the energy and cross sections for charge transfer 
    from Belkic et al 1992 Phys. Rev. A. 80, Table 5. Essentially an update 
    to Shakeshaft 1978.
   
    Cross sections are of collisions between protons and neutral H to 

    1s : Q_p1
    2s : Q_p_2s
    2p : Q_p_2p
    n=2: sum of 2s and 2p states, Q_p2
    3s : Q_p_3s
    3p : Q_p_3p
    3d : Q_p_3d
    n=3 : sum of 3s, 3p, 3d states, Q_p3
 
    Energy range  1 -- 100 keV

    '''
    def __init__(self):
        self.energy = np.array((40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 125.0, 150.0, 
                                200.0, 300.0, 400, 500.0, 600.0, 700., 800.0, 900.0, 1000.0))
        self.Q_p1 = np.array((1.37e-16, 6.95e-17, 3.87e-17, 2.30e-17, 1.45e-17, 9.45e-18, 6.39e-18, 2.70e-18, 1.29e-18,
                              3.75e-19, 5.89e-20, 1.47e-20, 4.83e-21, 1.90e-21, 8.55e-22, 4.24e-22, 2.27e-22, 1.29e-22))*1e17
        self.Q_p_2s = np.array((2.81e-17, 1.44e-17, 8.00e-18, 4.71e-18, 2.91e-18, 1.87e-18, 1.24e-18, 5.05e-19, 2.33e-19,
                                6.43e-20, 9.38e-21, 2.24e-21, 7.13e-22, 2.75e-22, 1.22e-22, 5.96e-23, 3.16e-23, 1.79e-23))*1e17
        self.Q_p_2p = np.array((1.12e-17, 5.70e-18, 3.08e-18, 1.75e-18, 1.05e-18, 6.48e-19, 4.15e-19, 1.54e-19, 6.51e-20,
                                1.55e-20, 1.80e-21, 3.63e-22, 1.01e-22, 3.53e-23, 1.43e-23, 6.52e-24, 3.25e-24, 1.74e-24))*1e17
        self.Q_p2 = np.array((3.94e-17, 2.01e-17, 1.11e-17, 6.46e-18, 3.96e-18, 2.52e-18, 1.66e-18, 6.59e-19, 2.98e-19,
                              7.98e-20, 1.12e-20, 2.60e-21, 8.14e-22, 3.10e-22, 1.36e-22, 6.61e-23, 3.49e-23, 1.96e-23))*1e17
        self.Q_p_3s = np.array((8.96e-18, 4.65e-18, 2.59e-18, 1.53e-18, 9.43e-19, 6.06e-19, 4.02e-19, 1.62e-19, 7.41e-20,
                                2.03e-20, 2.91e-21, 6.88e-22, 2.18e-22, 8.38e-23, 3.69e-23, 1.18e-23, 9.55e-24, 5.39e-24))*1e17
        self.Q_p_3p = np.array((3.85e-18, 2.01e-18, 1.11e-18, 6.39e-19, 3.84e-19, 2.39e-19, 1.54e-19, 5.72e-20, 2.43e-20,
                                5.49e-21, 6.72e-22, 1.36e-22, 3.80e-23, 1.32e-23, 5.38e-24, 2.46e-24, 1.23e-24, 6.60e-25))*1e17
        self.Q_p_3d = np.array((1.02e-18, 4.39e-19, 2.11e-19, 1.10e-19, 6.05e-20, 3.51e-20, 2.13e-20, 7.04e-21, 2.74e-21,
                                5.82e-22, 6.05e-23, 1.18e-23, 3.33e-24, 1.18e-24, 4.95e-25, 2.33e-25, 1.20e-25, 6.67e-26))*1e17
        self.Q_p3 = np.array((1.38e-17, 7.10e-18, 3.91e-18, 2.28e-18, 1.39e-18, 8.80e-19, 5.77e-19, 2.26e-19, 1.01e-19,
                              2.66e-20, 3.64e-21, 8.35e-22, 2.59e-22, 9.82e-23, 4.28e-23, 2.07e-23, 1.09e-23, 6.11e-24))*1e17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

       

def cs_polyfit(energy, csec, emin = -100.0, emax=-100.0, 
               order = 3, log10E=False, log10Q = False):

    if emax == -100.0:
        emax = energy[-1]
    if emin == -100.0:
        emin = energy[0]

    eind1 = np.where(energy >= emin)[0][0]
    eind2 = np.where(energy <= emax)[0][-1]+1

    if log10E == True:
        energy = np.log10(energy[eind1:eind2])
    else:
        energy = energy[eind1:eind2]

    if log10Q == True:
        csec = np.log10(csec[eind1:eind2])
    else:
        csec = csec[eind1:eind2]


    pfit = Poly.fit(energy, csec, order)
    pfit_vals = Poly(pfit.convert().coef)    

    return pfit_vals


def cs_chebfit(energy, csec, emin = -100.0, emax=-100.0, 
               order = 3, log10E=False, log10Q = False):

    if emax == -100.0:
        emax = energy[-1]
    if emin == -100.0:
        emin = energy[0]

    eind1 = np.where(energy >= emin)[0][0]
    eind2 = np.where(energy <= emax)[0][-1]+1

    if log10E == True:
        energy = np.log10(energy[eind1:eind2])
    else:
        energy = energy[eind1:eind2]

    if log10Q == True:
        csec = np.log10(csec[eind1:eind2])
    else:
        csec = csec[eind1:eind2]


    # x = (np.log(energy/emin) - np.log(emax/energy))/np.log(emax/emin)

    cfit = Cheb.fit(energy,csec,order)
    # cfit = Cheb.fit(x, csec, order)
    cfit_vals = Cheb(cfit.convert().coef)

    return cfit_vals

def cs_hyberbolfit(energy, csec, emin = -100.0, emax = -100.0,
                   boundsin = [0, 1.1, 0, 1, 0, np.inf]):
    
    if emax == -100.0:
        emax = energy[-1]
    if emin == -100.0:
        emin = energy[0]

    eind1 = np.where(energy >= emin)[0][0]
    eind2 = np.where(energy <= emax)[0][-1]+1

    energy = energy[eind1:eind2]
    csec = csec[eind1:eind2]

    y0_guess = csec[0] 

    popt_hyp, pcov_hyp=curve_fit(hyperbolic_fn, 
                             energy, csec, 
                             bounds=[[y0_guess*boundsin[0],boundsin[2],boundsin[4]],
                                     [y0_guess*boundsin[1],boundsin[3],boundsin[5]]])

    # yvals = OZpy.CrossSections.hyperbolic_fn(xvals, 
                                  # *popt_hyp)

    return popt_hyp, pcov_hyp

def cs_expfit(energy, csec, emin = -100.0, emax = -100.0,
              boundsin = [0.5,2.5,-10,10]):
    
    if emax == -100.0:
        emax = energy[-1]
    if emin == -100.0:
        emin = energy[0]

    eind1 = np.where(energy >= emin)[0][0]
    eind2 = np.where(energy <= emax)[0][-1]+1

    energy = energy[eind1:eind2]
    csec = csec[eind1:eind2]

    y0_guess = csec[0] 


    popt_exp, pcov_exp=curve_fit(exponential_fn, 
                             energy, csec,
                             bounds=[[y0_guess*boundsin[0],boundsin[2]],[y0_guess*boundsin[1],boundsin[3]]])

    #yvals = OZpy.CrossSections.exponential_fn(xvals, 
                                  # *popt_exp)

    return popt_exp, pcov_exp


def hyperbolic_fn(x, y0, b, b0):
    '''
    Hyperbolic decline curve equation
    Arguments:
        x: Float. 
           Energy, usually in keV
        y0: Float. 
            Initial value of y at start of curve.
        b: Float. 
            Hyperbolic decline constant
        b0: Float. 
            Nominal decline rate at time x=0
    Output: 
        q: Float.
           The value of the function at value x
    '''
    return y0/((1.0+b*b0*x)**(1.0/b))


def exponential_fn(x, y0, b):
    """
    Exponential decline curve equation
    Arguments:
         x: Float. 
           Energy, usually in keV
        y0: Float. 
            Initial value of y at start of curve.
        b: Float. 
            Decline constant
    Output: 
        q: Float.
           The value of the function at value x
    """
    return y0*np.exp(-b*x)
 
