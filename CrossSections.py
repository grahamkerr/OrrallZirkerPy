import numpy as np 


class CrossSec:

    def __init__(self, energy):
        self.energy = energy


    def cs_fang95(self
    	):

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
            A dictionary containing each cross section, in 10^-17 cm^2 


        Graham Kerr
        July 2021

        '''

        ########################################################################
        # Some preliminary set up
        ########################################################################
 
        ## Turn to list if an integer or float are provided
        if type(self.energy) == int:
            self.energy = [self.energy]
        if type(self.energy) == float:
            self.energy = [self.energy]

        ## How many energies to calculate
        nE = len(self.energy)
        
        Q_p1 = np.zeros([nE],dtype = np.float64)
        Q_p2 = np.zeros([nE],dtype = np.float64)
        Q_p3 = np.zeros([nE],dtype = np.float64)
        Q_1pP = np.zeros([nE],dtype = np.float64)
        Q_1pH = np.zeros([nE],dtype = np.float64)
        Q_1pE = np.zeros([nE],dtype = np.float64)
        Q_12P = np.zeros([nE],dtype = np.float64)
        Q_12H = np.zeros([nE],dtype = np.float64)
        Q_12E = np.zeros([nE],dtype = np.float64)
        Q_13P = np.zeros([nE],dtype = np.float64)
        Q_13H = np.zeros([nE],dtype = np.float64)
        Q_13E = np.zeros([nE],dtype = np.float64)
        Q_23E = np.zeros([nE],dtype = np.float64)

        ########################################################################
        # Go through each energy and calculate the cross section
        ########################################################################

        for ind in range(nE):

            print(self.energy[ind])
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
                a0,a1,a2,a3,a4 = 1.1452, -1.33788, 3.9887e-1, 3.6992e-2, 1.0810e-3
                Q_p3[ind] = a0 + (a1*self.energy[ind]**1) + (a2*self.energy[ind]**2) + (a3*self.energy[ind]**3) + (a4*self.energy[ind]**4)
            elif self.energy[ind] > 16.0:
                a0,a1,a2,a3 = 2.0134, -0.03773, 2.500e-4, -4.3204e-7
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

        # out  = {'Q_p1':Q_p1,
        #              'Q_p2':Q_p2,
        #              'Q_p3':Q_p3,
        #              'Q_1pP':Q_1pP,
        #              'Q_1pH':Q_1pH,
        #              'Q_1pE':Q_1pE,
        #              'Q_12P':Q_12P,
        #              'Q_12H':Q_12H,
        #              'Q_12E':Q_12E,
        #              'Q_13P':Q_13P,
        #              'Q_13H':Q_13H,
        #              'Q_13E':Q_13E,
        #              'Q_23E':Q_23E,
        #              'self.energy':self.energy,
        #              'Units':'energy in [keV], Q in [10^-17 cm^-2]'}

        # out = cs_fang95_out(Q_p1, Q_p2, Q_p3, 
        #                     Q_1pP, Q_1pH, Q_1pE,
        #                     Q_12P, Q_12H, Q_12E,
        #                     Q_13P, Q_13H, Q_13E,
        #                     Q_23E)
        out = cs_fang95_out()

        return out

