import numpy as np 
from numpy.polynomial.polynomial import Polynomial as Poly
from scipy.optimize import curve_fit

################################################################################
################################################################################
################################################################################

class CrossSecH:

    '''

    Cross sections for Hydrogen model atom

    Inputs
    _______

    energy -- projectile energy in keV (single value or array)

    Methods
    _______
     
    The methods attached to this class take the input energy and return various 
    cross sections (see each method for a comprehensive list). Methods are:

        kerr_fit_poly -- Mostly 8-degree polynomial fits to the underlying data 
                          held in CrossSections.py. The higher energy ranges are 
                          fit with straight lines in logE-logQ space to extrapolate 
                          past the underlying data (use with caution). 
                          Also includes several functional forms from the IAEA Vol 4 
                          (Janev et al 1993).
        kerr_fit_cheb -- Mostly 8-degree Chebyshev fits to the underlying data 
                          held in CrossSections.py. The higher energy ranges are 
                          fit with straight lines in logE-logQ space to extrapolate 
                          past the underlying data (use with caution). 
                          Also includes several functional forms from the IAEA Vol 4 
                          (Janev et al 1993).
        fang95 -- The fits from Fang, Feautrier & Henoux et al 1995 A&A 297, 854. 
                  Fang et al 1995 did not detail the energy bounds of the fits, so 
                  use these with caution. They go negative in lots of places at 
                  100 keV (or much lower for electron impacts).
        bw99 -- The fits from Brosius & Woodgate 1999, ApJ 514. These fits use older data
                so there are some big differences between the fits to more modern data in 
                kerr_fit_XXXX

        Each of these outputs an object with energy in keV and cross sections in 10^-17 cm^2.



    Notes
    ______

    This class takes a projectile energy in keV, either a single value or a list, 
    and has various methods to compute cross sections of charge exchange or impact 
    excitation/ionisation interactions between that energetic particle with ambient 
    particles.

    The BW99 and Fang95 fits are for reference. 

    The others are my own fits, combined with fits from sources such as IAEA. 

    The underlying data for each cross section has different valid energy ranges, 
    and so the fits have been made to those ranges, and have been extended 
    via fitting linear decays (in logE-logQ space) to go to arbrotarily high energy 
    (though of course those extrapolations should be used with care). 

    Generally I would try and keep to 1 keV to 1 MeV. 

    The underlying data are held in seperate classes within this script, and the fitting
    functions are also located in this script. References are scattered throughout.



    Original:  
    - Graham Kerr, July 2021
    
    Modified (only major mods listed):
    - Graham Kerr, August 7th 2021, Started adding cross sections for suprathermal helium, 
      so renamed this module CrossSecH.py.

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

################################################################################

    def cs_kerr_cheb(self):
        '''
        This function will calculate the cross-sections required 
        to compute the population of suprathermal neutral hydrogen,
        given an energy E in keV. 

        Those cross sections are: 

        Q_pj -- the charge exchange cs from protons to H level j = 1,2

        The following cross sections are computed:

        Q_p1 -> charge exchange to n=1 (ground)
        Q_p2 -> charge exchange to n=2 
        Q_p3 -> charge exchange to n=3
        Q_p_3s -> charge exchange to 3s
        Q_p_3p -> charge exchange to 3p
        Q_p_3d -> charge exchange to 3d
        Q_p_2s -> charge exchange to 2s
        Q_p_2p -> charge exchange to 2p

        CX are are 8-degree Chebyshev fits to Winters et al 2009 (1-100keV) 
        and Belkic et al 1992 (125-8000) keV

        Q_1pP is 8-deg Chebyshev fit to Shah et al 1981,1987a,1998, covering
        the range 1.25-1500 keV. Since the fit is in log-log space a straight line
        is fit between 500-1500 keV, and any energy > 1500 keV is extrapolated as 
        a linear decay following the 8-deg polynomial fit

        Q_1pH is an 8-deg Chebyshev it to data in Cariatore & Schultz 2021, ApJS 

        Several cross sections are from the IAEA suppl series Vol 4, edited by Janev. 
        https://inis.iaea.org/collection/NCLCollectionStore/_Public/25/024/25024274.pdf?r=1
        It collates many sources. I think that it offers the best way to get electron 
        impact cross sections, and proton impact cross sections to/from excited states, 
        as I have been unable to find those elsewhere in convenient ways.
        They are functional forms, that I have checked extend well to energies higher
        than quoted (so I dont need to fit linear decay in log-log). These are marked with 
        an asterix **.


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

        Fits were done in linear (energy) - ln(cross-sec) space, 
        with energy in keV and cross sec in 10^-17 cm^2
        


        Graham Kerr
        July 2021

        '''
        ########################################################################
        # Some preliminary set up
        ########################################################################

    
        emin_cx = 1.0
        emax_cx = 8000.0

        ########################################################################
        # Calculate the cross sections from the coeff.
        ########################################################################

        # for ind in range(self.nE):

        ########
        # Q_p1
        ########
        coefs_qp1 = np.array((-9.02130231e+00, -1.41956219e+01, -4.03526188e+00,  2.65162821e-01,
                              4.63966058e-01, -1.41408151e-01, -7.16650392e-02,  7.50934329e-02,
                              1.21348395e-03))
        deg = len(coefs_qp1)
        Q_p1 = np.exp(chebyshev_fn(self.energy, coefs_qp1, deg, emin_cx, emax_cx))

        ########
        # Q_p2
        ########
        coefs_qp2 = np.array((-13.33234177, -13.09534279,  -4.50454333,   0.37329088,
                              0.73414779,  -0.27266491,  -0.14132532,   0.16552985,
                              -0.02563817))
        deg = len(coefs_qp2)
        Q_p2 = np.exp(chebyshev_fn(self.energy, coefs_qp2, deg, emin_cx, emax_cx))

        ########
        # Q_p2s
        ########
        coefs_qp2s = np.array((-1.64201279e+01, -1.13951670e+01, -5.92106146e+00,  7.35532224e-01,
                               7.21043358e-01, -3.94619891e-01, -5.07871953e-02,  1.43313562e-01,
                               -1.24757704e-02))
        deg = len(coefs_qp2s)
        Q_p_2s = np.exp(chebyshev_fn(self.energy, coefs_qp2s, deg, emin_cx, emax_cx))

        ########
        # Q_p2p
        ########
        coefs_qp2p = np.array((-16.63011227, -14.65902507,  -4.62838587,   0.49142988,
                               0.68755281,  -0.15925478,  -0.10527213,   0.06216281,
                               -0.0366277 ))
        deg = len(coefs_qp2p)
        Q_p_2p = np.exp(chebyshev_fn(self.energy, coefs_qp2p, deg, emin_cx, emax_cx))

        ########
        # Q_p3
        ########
        coefs_qp3 = np.array((-1.65845503e+01, -1.20343070e+01, -5.05549945e+00,  6.46012371e-01,
                               6.87680549e-01, -2.52531499e-01, -1.70099272e-01,  1.71261267e-01,
                               2.32330065e-04))
        deg = len(coefs_qp3)
        Q_p3 = np.exp(chebyshev_fn(self.energy, coefs_qp3, deg, emin_cx, emax_cx))

        ########
        # Q_p3s
        ########
        coefs_qp3s = np.array((-19.33429628, -10.19936388,  -5.86496749,   0.5161523 ,
                               1.23253673,  -0.66462294,  -0.15508603,   0.36056999,
                               -0.19231693))
        deg = len(coefs_qp3s)
        Q_p_3s = np.exp(chebyshev_fn(self.energy, coefs_qp3s, deg, emin_cx, emax_cx))

        ########
        # Q_p3p
        ########
        coefs_qp3p = np.array((-20.48185934, -13.05831177,  -5.49675648,   0.73823815,
                               0.75441546,  -0.24984525,  -0.09987703,   0.10492206,
                               -0.0594819 ))
        deg = len(coefs_qp3p)
        Q_p_3p = np.exp(chebyshev_fn(self.energy, coefs_qp3p, deg, emin_cx, emax_cx))

        ########
        # Q_p3d
        ########
        coefs_qp3d = np.array((-2.36013894e+01, -1.41823641e+01, -5.04558039e+00,  1.07973772e+00,
                               3.03491139e-01, -5.46681975e-02, -5.36399104e-02, -1.17651735e-01,
                               2.03595395e-02))
        deg = len(coefs_qp3d)
        Q_p_3d = np.exp(chebyshev_fn(self.energy, coefs_qp3d, deg, emin_cx, emax_cx))


        #######
        # Q_1pP
        #######
        emin = 1.25
        emax = 1500
        coefs_q1pP_1 = np.array((1.40779597,  1.80739461, -1.42774774,  0.01782048,  0.35604087,
                                 -0.06343798, -0.10090822,  0.03552779,  0.03316916))
        coefs_q1pP_2 = np.array((2.76008009, -0.82241237))
        deg = len(coefs_q1pP_1)
        polfit_2 = Poly(coefs_q1pP_2)
        if np.max(self.energy > 1500):
            eind1 = np.where(self.energy <= 1500)[0][-1]
            Q_1pP_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q1pP_1, deg, emin, emax))
            Q_1pP_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_1pP = np.concatenate([Q_1pP_tmp1,Q_1pP_tmp2])
        else: 
            Q_1pP = np.exp(chebyshev_fn(self.energy, coefs_q1pP_1, deg, emin, emax))

        #######
        # Q_1pH
        #######
        emin = 1.00
        emax = 1e4
        coefs_q1pH = np.array((0.2535636 , -1.18711515, -1.57676689,  0.30765838,  0.11554446,
                               -0.15792788,  0.03235062,  0.06139648, -0.05831816)) 
        deg = len(coefs_q1pH)
        Q_1pH = np.exp(chebyshev_fn(self.energy, coefs_q1pH, deg, emin, emax))

        ######H
        # Q_1pE
        #######
        eth = 13.6
        A = 0.18450
        B = np.array((-0.032226, -0.0343539, 1.4003, -2.8115, 2.2986))
        Q_1pE = A*np.log((self.energy*1e3)/eth)
        for i in range(len(B)):
            Q_1pE+= B[i]*(1 - eth/(self.energy*1e3))**i
        Q_1pE = Q_1pE*1e-13/(eth*(self.energy*1e3))*1e17
       
        ######H
        # Q_1pE (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space
        emin = 0.015
        emax = 10
        coefs_q1pE_alt_1 = np.array((1.32318098, -1.47101701, -0.41715532,  0.46103678, -0.23126487,
                                 0.13416711, -0.09528776,  0.0616035 , -0.03513007))
        coefs_q1pE_alt_2 = np.array((0.06223376, -0.87223181))
        deg = len(coefs_q1pE_alt_1)
        polfit_2 = Poly(coefs_q1pE_alt_2)
        if np.max(self.energy > 10):
            eind1 = np.where(self.energy <= 10)[0][-1]
            Q_1pE_alt_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q1pE_alt_1, deg, emin, emax))
            Q_1pE_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_1pE_alt = np.concatenate([Q_1pE_alt_tmp1,Q_1pE_alt_tmp2])
        else: 
            Q_1pE_alt = np.exp(chebyshev_fn(self.energy, coefs_q1pE_alt_1, deg, emin, emax))

        #######
        # Q_12P **
        #######
        amu = 1.00797
        A = np.array((34.433, 44.507, 0.56870, 8.5476, 7.8501, 
                      -9.2217, 1.8020e-2, 1.6931, 1.9422e-3, 2.9067))
        val1 = (np.exp(-A[1]/(self.energy/amu))*np.log(1+A[2]*(self.energy/amu)))/(self.energy/amu)
        val2 = (A[3]*np.exp(-1.0*A[4]*(self.energy/amu)))/((self.energy/amu)**A[5])
        val3 = (A[6]*np.exp(-1.0*A[7]/(self.energy/amu)))/(1+A[8]*((self.energy/amu)**A[9]))
        Q_12P = 1e-16*A[0]*(val1+val2+val3)*1e17

        ######H
        # Q_12P (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space
        emin = 6e2/1e3
        emax = 5e6/1e3
        coefs_q12P_alt_1 = np.array((4.575172  , -0.63998318,  0.17872151, -0.34091127, -0.26607579,
                                     0.11983473, -0.33773698, -0.04716123,  0.04230359))
        coefs_q12P_alt_2 = np.array((2.74816921, -0.81042653))
        deg = len(coefs_q12P_alt_1)
        polfit_2 = Poly(coefs_q12P_alt_2)
        if np.max(self.energy > 5e3):
            eind1 = np.where(self.energy <= 5e3)[0][-1]
            Q_12P_alt_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q12P_alt_1, deg, emin, emax))
            Q_12P_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_12P_alt = np.concatenate([Q_12P_alt_tmp1,Q_12P_alt_tmp2])
        else: 
            Q_12P_alt = np.exp(chebyshev_fn(self.energy, coefs_q12P_alt_1, deg, emin, emax))


        #######
        # Q_12H
        #######
        emin = 1
        emax = 100
        coefs_q12H_1 = np.array((2.06452447e+00, -2.14833594e-01, -7.77200967e-01,  2.70225749e-01,
                                -1.65983078e-03, -2.47692743e-02,  5.42648554e-03, -1.34736013e-02,
                                1.44486473e-03))
        coefs_q12H_2 = np.array((1.35631047, -0.6056955))
        deg = len(coefs_q12H_1)
        polfit_2 = Poly(coefs_q12H_2)
        if np.max(self.energy > 100):
            eind1 = np.where(self.energy <= 100)[0][-1]
            Q_12H_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q12H_1, deg, emin, emax))
            Q_12H_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_12H = np.concatenate([Q_12H_tmp1,Q_12H_tmp2])
        else: 
            Q_12H = np.exp(chebyshev_fn(self.energy, coefs_q12H_1, deg, emin, emax))

        #######
        # Q_12H (2s)
        #######
        emin = 2
        emax = 100
        coefs_q12H_2s_1 = np.array((-0.68769663, -0.21757511, -0.46717881, -0.0525721 ,  0.08776152,
                                    -0.00188545, -0.01192942, -0.02846297,  0.04375735))
        coefs_q12H_2s_2 = np.array((1.2894321 , -0.88626221))
        deg = len(coefs_q12H_2s_1)
        polfit_2 = Poly(coefs_q12H_2s_2)
        if np.max(self.energy > 100):
            eind1 = np.where(self.energy <= 100)[0][-1]
            Q_12H_2s_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q12H_2s_1, deg, emin, emax))
            Q_12H_2s_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_12H_2s = np.concatenate([Q_12H_2s_tmp1,Q_12H_2s_tmp2])
        else: 
            Q_12H_2s = np.exp(chebyshev_fn(self.energy, coefs_q12H_2s_1, deg, emin, emax))

        #######
        # Q_12H (2p)
        #######
        emin = 1
        emax = 100
        coefs_q12H_2p_1 = np.array((1.42684893, -0.19496454, -0.89427881,  0.42995908, -0.04699293,
                                    -0.04457275,  0.00947055, -0.01001742,  0.01104788))
        coefs_q12H_2p_2 = np.array((1.03055114, -0.50060726))
        deg = len(coefs_q12H_2p_1)
        polfit_2 = Poly(coefs_q12H_2p_2)
        if np.max(self.energy > 100):
            eind1 = np.where(self.energy <= 100)[0][-1]
            Q_12H_2p_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q12H_2p_1, deg, emin, emax))
            Q_12H_2p_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_12H_2p = np.concatenate([Q_12H_2p_tmp1,Q_12H_2p_tmp2])
        else: 
            Q_12H_2p = np.exp(chebyshev_fn(self.energy, coefs_q12H_2p_1, deg, emin, emax))

        ####### 
        # Q_12E **
        #######
        ## The below is only for E > 12.23 eV (which for us is fine)
        eth = 10.2
        A = np.array((1.4182, -20.877, 49.735, -46.249, 17.442))
        B = 4.4979
        Q_12E = 0
        for i in range(len(A)):
            Q_12E+= A[i]/((self.energy*1e3/eth)**i)  
        Q_12E+= B*np.log(self.energy*1e3/eth)
        Q_12E = Q_12E*5.984e-16/(eth*(self.energy*1e3/eth))*1e17
       
        ######H
        # Q_12E (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space
        emin = 1.02e1/1e3
        emax = 1e4/1e3
        coefs_q12E_alt_1 = np.array((8.47065482e-01, -1.57769047e+00, -7.79759234e-01,  2.87835879e-01,
                                    -7.98831339e-02,  6.76667688e-03,  7.56566128e-03, -4.81387004e-03,
                                    1.06774858e-0))
        coefs_q12E_alt_2 = np.array((0.12140192, -0.82993883))
        deg = len(coefs_q12E_alt_1)
        polfit_2 = Poly(coefs_q12E_alt_2)
        if np.max(self.energy > 10):
            eind1 = np.where(self.energy <= 10)[0][-1]
            Q_12E_alt_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q12E_alt_1, deg, emin, emax))
            Q_12E_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_12E_alt = np.concatenate([Q_12E_alt_tmp1,Q_12E_alt_tmp2])
        else: 
            Q_12E_alt = np.exp(chebyshev_fn(self.energy, coefs_q12E_alt_1, deg, emin, emax))

        #######
        # Q_13P **
        #######
        amu = 1.00797
        A = np.array((6.1950, 35.773, 0.54818, 5.5162e-3,
                      0.29114, -4.5264, 6.0311, -2.0679))
        val1 = (np.exp(-1*A[1]/(self.energy/amu)) * np.log(1+A[2]*(self.energy/amu)))
        val2 = (A[3]*np.exp(-1*A[4]*(self.energy/amu)))/(((self.energy/amu)**A[5]) + A[6]*((self.energy/amu)**A[7]))
        Q_13P = 1e-16*A[0]*(val1+val2)*1e17

        ######H
        # Q_13P (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space
        emin = 5e2/1e3
        emax = 5e6/1e3
        coefs_q13P_alt_1 = np.array((-2.41300466e+00,  1.01745307e+00, -1.67949789e+00,  2.90622063e-01,
                                     6.45008918e-02,  7.37360354e-02, -8.66650713e-02, -1.40441847e-04,
                                     8.00890956e-02))
        coefs_q13P_alt_2 = np.array((1.98572466, -0.80690771))
        deg = len(coefs_q13P_alt_1)
        polfit_2 = Poly(coefs_q13P_alt_2)
        if np.max(self.energy > 5e3):
            eind1 = np.where(self.energy <= 5e3)[0][-1]
            Q_13P_alt_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q13P_alt_1, deg, emin, emax))
            Q_13P_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_13P_alt = np.concatenate([Q_13P_alt_tmp1,Q_13P_alt_tmp2])
        else:
            Q_13P_alt = np.exp(chebyshev_fn(self.energy, coefs_q13P_alt_1, deg, emin, emax))

        #######
        # Q_13H
        #######
        emin = 1
        emax = 1024
        coefs_q13H_1 = np.array((-2.42745772, -1.36363505, -1.3146549 ,  0.54569869, -0.28706635,
                                 0.02859126,  0.05985655, -0.02142932, -0.01579549))
        coefs_q13H_2 = np.array((1.49527064, -0.92320445))
        deg = len(coefs_q13H_1)
        polfit_2 = Poly(coefs_q13H_2)
        if np.max(self.energy > 1024):
            eind1 = np.where(self.energy <= 1024)[0][-1]
            Q_13H_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q13H_1, deg, emin, emax))
            Q_13H_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_13H = np.concatenate([Q_13H_tmp1,Q_13H_tmp2])
        else: 
            Q_13H = np.exp(chebyshev_fn(self.energy, coefs_q13H_1, deg, emin, emax))

        #######
        # Q_13H (3s)
        #######
        emin = 1
        emax = 1024
        coefs_q13H_3s_1 = np.array((-4.67042235, -2.06751673, -1.07787458,  0.52013411, -0.26339177,
                                    0.05823117,  0.03928287, -0.03383436, -0.00482211))
        coefs_q13H_3s_2 = np.array((0.87560595, -0.97693724))
        deg = len(coefs_q13H_3s_1)
        polfit_2 = Poly(coefs_q13H_3s_2)
        if np.max(self.energy > 1024):
            eind1 = np.where(self.energy <= 1024)[0][-1]
            Q_13H_3s_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q13H_3s_1, deg, emin, emax))
            Q_13H_3s_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_13H_3s = np.concatenate([Q_13H_3s_tmp1,Q_13H_3s_tmp2])
        else:
            Q_13H_3s = np.exp(chebyshev_fn(self.energy, coefs_q13H_3s_1, deg, emin, emax))

        #######
        # Q_13H (3p)
        #######
        emin = 1
        emax = 1024
        coefs_q13H_3p_1 = np.array((-3.56654351, -0.97121545, -1.52139543,  0.62693773, -0.32563846,
                                    0.02920606,  0.06675371, -0.0205075 , -0.01750213))
        coefs_q13H_3p_2 = np.array((1.3347965 , -0.90787652))
        deg = len(coefs_q13H_3p_1)
        polfit_2 = Poly(coefs_q13H_3p_2)
        if np.max(self.energy > 1024):
            eind1 = np.where(self.energy <= 1024)[0][-1]
            Q_13H_3p_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q13H_3p_1, deg, emin, emax))
            Q_13H_3p_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_13H_3p = np.concatenate([Q_13H_3p_tmp1,Q_13H_3p_tmp2])
        else:
            Q_13H_3p = np.exp(chebyshev_fn(self.energy, coefs_q13H_3p_1, deg, emin, emax))

        #######
        # Q_13H (3d)
        #######
        emin = 1
        emax = 1024
        coefs_q13H_3d_1 = np.array((-8.90572044, -0.64674584, -1.9169128 ,  0.74557654, -0.36485015,
                                    0.03666598,  0.08664943, -0.0292461 , -0.02377451))
        coefs_q13H_3d_2 = np.array((0.42432161, -0.959331822))
        deg = len(coefs_q13H_3d_1)
        polfit_2 = Poly(coefs_q13H_3d_2)
        if np.max(self.energy > 1024):
            eind1 = np.where(self.energy <= 1024)[0][-1]
            Q_13H_3d_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q13H_3d_1, deg, emin, emax))
            Q_13H_3d_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_13H_3d = np.concatenate([Q_13H_3d_tmp1,Q_13H_3d_tmp2])
        else:
            Q_13H_3d = np.exp(chebyshev_fn(self.energy, coefs_q13H_3d_1, deg, emin, emax))


        #######
        # Q_13E **
        #######
        eth = 12.09
        A = np.array((0.42956, -0.58288, 1.0693, 0.0))
        B = 0.75488
        C = 0.38277
        Q_13E = 0
        for i in range(len(A)):
            Q_13E+= A[i]/((self.energy*1e3/eth)**i)
        Q_13E+= B*np.log(self.energy*1e3/eth)
        Q_13E = Q_13E * ((self.energy*1e3 - eth)/(self.energy*1e3))**C
        Q_13E = Q_13E * 5.984e-16/(eth*(self.energy*1e3/eth))*1e17

        ######H
        # Q_13E (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space
        emin = 1.24e1/1e3
        emax = 1e4/1e3
        coefs_q13E_alt_1 = np.array((-0.39785756, -1.59506388,  0.24481796,  0.54878841, -0.38718707,
                                     0.29794288, -0.21806939,  0.14719962, -0.08219712))
        coefs_q13E_alt_2 = np.array((-0.6484916 , -0.82911879))
        deg = len(coefs_q13E_alt_1)
        polfit_2 = Poly(coefs_q13E_alt_2)
        if np.max(self.energy > 10):
            eind1 = np.where(self.energy <= 10)[0][-1]
            Q_13E_alt_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q13E_alt_1, deg, emin, emax))
            Q_13E_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_13E_alt = np.concatenate([Q_13E_alt_tmp1,Q_13E_alt_tmp2])
        else: 
            Q_13E_alt = np.exp(chebyshev_fn(self.energy, coefs_q13E_alt_1, deg, emin, emax))

        #######
        # Q_2pP **
        #######
        amu = 1.00797
        A = np.array((107.63, 29.860, 1.0176e6, 6.9713e-3, 
                      2.8488e-2, -1.8000, 4.7852e-2, -0.20923))
        val1 = (np.exp(-1*A[1]/(self.energy/amu)) * np.log(1+A[2]*(self.energy/amu)))/(self.energy/amu)
        val2 = (A[3]*np.exp(-1*A[4]*(self.energy/amu)))/((self.energy/amu)**A[5] + A[6]*(self.energy/amu)**A[7])
        Q_2pP = 1e-16*A[0]*(val1+val2)*1e17

        ######H
        # Q_2pP (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space
        emin = 5e2/1e3
        emax = 5e6/1e3
        coefs_q2pP_alt_1 = np.array((6.85584607,  0.08379775, -1.92823987,  0.40025618,  0.2571304 ,
                                    -0.14461441, -0.05612802,  0.09011405,  0.00924564))
        coefs_q2pP_alt_2 = np.array((4.07769286, -0.917189))
        deg = len(coefs_q2pP_alt_1)
        polfit_2 = Poly(coefs_q2pP_alt_2)
        if np.max(self.energy > 5e3):
            eind1 = np.where(self.energy <= 5e3)[0][-1]
            Q_2pP_alt_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q2pP_alt_1, deg, emin, emax))
            Q_2pP_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_2pP_alt = np.concatenate([Q_2pP_alt_tmp1,Q_2pP_alt_tmp2])
        else:
            Q_2pP_alt = np.exp(chebyshev_fn(self.energy, coefs_q2pP_alt_1, deg, emin, emax))

        #######
        # Q_2pH
        #######

        #######
        # Q_2pE **
        #######
        eth = 3.4
        A = 0.14784
        B = np.array((0.0080871, -0.062270, 1.9414, -2.1980, 0.9584))
        Q_2pE = A*np.log((self.energy*1e3)/eth)
        for i in range(len(B)):
            Q_2pE+= B[i]*(1 - eth/(self.energy*1e3))**i
        Q_2pE = Q_2pE*1e-13/(eth*(self.energy*1e3))*1e17

        #######
        # Q_2pE (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space
        emin = 4e0/1e3
        emax = 1e4/1e3
        coefs_q2pE_alt_1 = np.array(( 5.60146536, -2.51283518, -0.28192902,  0.45285694, -0.25986116,
                                      0.1547214 , -0.09008906,  0.0471847 , -0.02968324))
        coefs_q2pE_alt_2 = np.array((0.63663501, -0.90108125))
        deg = len(coefs_q2pE_alt_1)
        polfit_2 = Poly(coefs_q2pE_alt_2)
        if np.max(self.energy > 10):
            eind1 = np.where(self.energy <= 10)[0][-1]
            Q_2pE_alt_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q2pE_alt_1, deg, emin, emax))
            Q_2pE_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_2pE_alt = np.concatenate([Q_2pE_alt_tmp1,Q_2pE_alt_tmp2])
        else: 
            Q_2pE_alt = np.exp(chebyshev_fn(self.energy, coefs_q2pE_alt_1, deg, emin, emax))


        #######
        # Q_23P **
        #######
        amu = 1.00797
        A = np.array((394.51, 21.606, 0.62426, 0.013596, 0.16565, -0.8949))
        val1 = (np.exp(A[1]/(self.energy/amu)) * np.log(1+A[2]*(self.energy/amu)))/(self.energy/amu)
        val2 = (A[3]*np.exp(-1.0*A[4]*(self.energy/amu)))/(((self.energy/amu))**A[5])
        Q_23P = 1e-16*A[0]*(val1+val2)*1e17

        ######H
        # Q_23P (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space
        emin = 5e2/1e3
        emax = 5e6/1e3
        coefs_q23P_alt_1 = np.array((7.83579797e+00, -8.19844851e-01, -1.10055603e+00,  9.33357070e-02,
                                    1.38898147e-01, -3.36799265e-02, -3.49725871e-02,  2.59337536e-02,
                                    2.59969409e-04))
        coefs_q23P_alt_2 = np.array((3.90949306, -0.8397776))
        deg = len(coefs_q23P_alt_1)
        polfit_2 = Poly(coefs_q23P_alt_2)
        if np.max(self.energy > 5e3):
            eind1 = np.where(self.energy <= 5e3)[0][-1]
            Q_23P_alt_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q23P_alt_1, deg, emin, emax))
            Q_23P_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_23P_alt = np.concatenate([Q_23P_alt_tmp1,Q_23P_alt_tmp2])
        else:
            Q_23P_alt = np.exp(chebyshev_fn(self.energy, coefs_q23P_alt_1, deg, emin, emax))


        #######
        # Q_23H
        #######

        #######
        # Q_23E **
        #######
        eth = 1.899
        A = np.array((5.2373, 119.25, -595.39, 816.71))
        B = 38.906
        C = 1.3196
        Q_23E = 0
        for i in range(len(A)):
            Q_23E+= A[i]/((self.energy*1e3/eth)**(i-1))
        Q_23E+= B*np.log(self.energy*1e3/eth)
        Q_23E = Q_23E * ((self.energy*1e3 - eth)/(self.energy*1e3))**C
        Q_23E = Q_23E * 5.984e-16/(eth*(self.energy*1e3/eth))*1e17

        ######H
        # Q_23E (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space
        emin = 2e0/1e3
        emax = 1e4/1e3
        coefs_q23E_alt_1 = np.array(( 6.88850867e+00, -2.60595372e+00, -9.15696581e-01,  2.35419562e-01,
                                     -4.85185197e-02, -4.14909074e-02,  6.15003585e-02, -3.85093036e-02,
                                      5.79295021e-03))
        coefs_q23E_alt_2 = np.array((1.17839148, -0.86544671))
        deg = len(coefs_q23E_alt_1)
        polfit_2 = Poly(coefs_q23E_alt_2)
        if np.max(self.energy > 10):
            eind1 = np.where(self.energy <= 10)[0][-1]
            Q_23E_alt_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q23E_alt_1, deg, emin, emax))
            Q_23E_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_23E_alt = np.concatenate([Q_23E_alt_tmp1,Q_23E_alt_tmp2])
        else: 
            Q_23E_alt = np.exp(chebyshev_fn(self.energy, coefs_q23E_alt_1, deg, emin, emax))

        #######
        # Q_3pP **
        #######
        amu = 1.00797
        A = np.array((326.26, 13.608, 4.9910e3, 3.0560e-1, 
                      6.4364e-2, -0.14924, 3.1525, -1.6314))
        val1 = (np.exp(-1*A[1]/(self.energy/amu))*np.log(1+A[2]*(self.energy/amu)))/(self.energy/amu)
        val2 = (A[3]*np.exp(-1*A[4]*(self.energy/amu)))/((self.energy/amu)**A[5]+A[6]*(self.energy/amu)**A[7])
        Q_3pP = 1e-16*A[0]*(val1 + val2)*1e17

        #######
        # Q_3pP (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space
        emin = 5e2/1e3
        emax = 5e6/1e3
        coefs_q3pP_alt_1 = np.array((9.8485377 , -1.43604878, -1.67213795,  0.48917483,  0.05845677,
                                     -0.14760691,  0.01898193,  0.05052536, -0.04183845))
        coefs_q3pP_alt_2 = np.array((4.46966962, -0.92106191))
        deg = len(coefs_q3pP_alt_1)
        polfit_2 = Poly(coefs_q3pP_alt_2)
        if np.max(self.energy > 5e3):
            eind1 = np.where(self.energy <= 5e3)[0][-1]
            Q_3pP_alt_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q3pP_alt_1, deg, emin, emax))
            Q_3pP_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_3pP_alt = np.concatenate([Q_3pP_alt_tmp1,Q_3pP_alt_tmp2])
        else:
            Q_3pP_alt = np.exp(chebyshev_fn(self.energy, coefs_q3pP_alt_1, deg, emin, emax))

        #######
        # Q_3pH
        #######

        #######
        # Q_3pE **
        #######
        eth = 1.511
        A = 0.058463
        B = np.array((-0.051272, 0.85310, -0.57014, 0.76684, 0.0))
        Q_3pE = A*np.log((self.energy*1e3)/eth)
        for i in range(len(B)):
            Q_3pE+= B[i]*(1 - eth/(self.energy*1e3))**i
        Q_3pE = Q_3pE*1e-13/(eth*(self.energy*1e3))*1e17

        #######
        # Q_3pE (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space
        emin = 2e0/1e3
        emax = 1e4/1e3
        coefs_q3pE_alt_1 = np.array(( 7.69844713, -3.12200882, -0.39146522,  0.41802356, -0.2297936 ,
                                      0.1255565 , -0.06042046,  0.02884291, -0.01828263))
        coefs_q3pE_alt_2 = np.array((0.95907927, -0.95892717))
        deg = len(coefs_q3pE_alt_1)
        polfit_2 = Poly(coefs_q3pE_alt_2)
        if np.max(self.energy > 10):
            eind1 = np.where(self.energy <= 10)[0][-1]
            Q_3pE_alt_tmp1 = np.exp(chebyshev_fn(self.energy[:eind1+1], coefs_q3pE_alt_1, deg, emin, emax))
            Q_3pE_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_3pE_alt = np.concatenate([Q_3pE_alt_tmp1,Q_3pE_alt_tmp2])
        else: 
            Q_3pE_alt = np.exp(chebyshev_fn(self.energy, coefs_q3pE_alt_1, deg, emin, emax))



        class cs_kerr_cheb_out:
            def __init__(selfout):
                selfout.Q_p1 = Q_p1
                selfout.Q_p2 = Q_p2
                selfout.Q_p_2s = Q_p_2s
                selfout.Q_p_2p = Q_p_2p
                selfout.Q_p3 = Q_p3
                selfout.Q_p_3s = Q_p_3s
                selfout.Q_p_3p = Q_p_3p
                selfout.Q_p_3d = Q_p_3d
                selfout.Q_1pP = Q_1pP
                selfout.Q_1pH = Q_1pH
                selfout.Q_1pE = Q_1pE
                selfout.Q_1pE_alt = Q_1pE_alt 
                selfout.Q_12P = Q_12P
                selfout.Q_12P_alt = Q_12P_alt
                selfout.Q_12H = Q_12H
                selfout.Q_12H_2s = Q_12H_2s
                selfout.Q_12H_2p = Q_12H_2p
                selfout.Q_12E = Q_12E
                selfout.Q_12E_alt = Q_12E_alt
                selfout.Q_13P = Q_13P
                selfout.Q_13P_alt = Q_13P_alt
                selfout.Q_13H = Q_13H
                selfout.Q_13H_3s = Q_13H_3s
                selfout.Q_13H_3p = Q_13H_3p
                selfout.Q_13H_3d = Q_13H_3d
                selfout.Q_13E = Q_13E
                selfout.Q_13E_alt = Q_13E_alt
                selfout.Q_2pP = Q_2pP
                selfout.Q_2pP_alt  = Q_2pP_alt
                # selfout.Q_2pH = Q_2pH
                selfout.Q_2pE = Q_2pE
                selfout.Q_2pE_alt = Q_2pE_alt
                selfout.Q_23P = Q_23P
                selfout.Q_23P_alt = Q_23P_alt
                # selfout.Q_23H = Q_23H
                selfout.Q_23E = Q_23E
                selfout.Q_23E = Q_23E_alt
                selfout.Q_3pP = Q_3pP
                selfout.Q_3pP_alt = Q_3pP_alt
                # selfout.Q_3pH = Q_3pH
                selfout.Q_3pE = Q_3pE
                selfout.Q_3pE_alt = Q_3pE_alt
                selfout.energy = self.energy
                selfout.Units = 'energy in [keV], Q in [10^-17 cm^-2]'

        out = cs_kerr_cheb_out()

        return out

################################################################################
  
    def cs_kerr_poly(self):
        '''
        This function will calculate the cross-sections required 
        to compute the population of suprathermal neutral hydrogen,
        given an energy E in keV. 

        Those cross sections are: 

        Q_pj -- the charge exchange cs from protons to H level j = 1,2

        The following cross sections are computed:

        Q_p1    -> charge exchange to n=1 (ground)
        Q_p2    -> charge exchange to n=2 
        Q_p3    -> charge exchange to n=3
        Q_p_3s -> charge exchange to 3s
        Q_p_3p -> charge exchange to 3p
        Q_p_3d -> charge exchange to 3d
        Q_p_2s -> charge exchange to 2s
        Q_p_2p -> charge exchange to 2p
        Q_1pP  -> proton impact ionisation from ground  
        Q_1pH  -> hydrogen impact ionisation from ground 
        Q_1pE  -> electron impact ionisation from ground 
        Q_12E  -> electron impact excitation from ground to n = 2
        Q_13E  -> electron impact excitation from ground to n = 3
        Q_2pE  -> electron impact ionisation from n=2
        Q_3pE  -> electron impact ionisation from n=3
        

        CX are are 8-degree polynomial fits to Winters et al 2009 (1-100keV) 
        and Belkic et al 1992 (125-8000) keV

        Q_1pP is 8-deg polynomial fit to Shah et al 1981,1987a,1998, covering
        the range 1.25-1500 keV. Since the fit is in log-log space a straight line
        is fit between 500-1500 keV, and any energy > 1500 keV is extrapolated as 
        a linear decay following the 8-deg polynomial fit

        Q_1pH is an 8-deg polynomial fit to data in cariatore & Schultz 2021, ApJS 

        Q_12H is an 8-deg polynomial fit to data in McLaughlin & Bell 1983 (36-100 keV), 
        and Hill et al 1979 (1-25 keV), with a linear fit to extrapolate to higher 
        energies. The data from 2s and 2p were summed to give the total.

        Q_13H is an 8-deg polynomial fit to data in McLaughlin & Bell 1987 (1-1024 keV), 
        with a linear fit to extrapolate to higher  energies. The data from 3s, 3p, abd 3d were 
        summed to give the total

        Several cross sections are from the IAEA suppl series Vol 4, edited by Janev. 
        https://inis.iaea.org/collection/NCLCollectionStore/_Public/25/024/25024274.pdf?r=1
        It collates many sources. I think that it offers the best way to get electron 
        impact cross sections, and proton impact cross sections to/from excited states, 
        as I have been unable to find those elsewhere in convenient ways.
        They are functional forms, that I have checked extend well to energies higher
        than quoted (so I dont need to fit linear decay in log-log). These are marked with 
        an asterix **.

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

        Fits were done with energy in keV, cross. sec in [10^-17 cm^2],
        in log-log space.

        The IAEA functions are need energy in eV.

        Graham Kerr
        July 2021

        '''
        ########################################################################
        # Calculate the cross sections from the fit coefficients
        ########################################################################

        ########
        # Q_p1
        ########
        coefs_qp1 = [2.22694265e+00,  2.19952462e-01, -2.61594140e+00,  4.88314799e+00,
                 -4.06254721e+00,  1.49092387e+00, -2.50756210e-01,  1.43541589e-02,
                 3.20664286e-04]
        polfit = Poly(coefs_qp1)

        Q_p1 = 10.00**(polfit(np.log10(self.energy)))


        ########
        # Q_p2
        ######## 
        coefs_qp2 = [ 3.52822513e-01,  1.64355656e+00, -7.98957810e+00,  1.55848788e+01,
                  -1.33785243e+01,  5.69351983e+00, -1.28933603e+00,  1.48436556e-01,
                  -6.77411207e-03]
        polfit = Poly(coefs_qp2)

        Q_p2 = 10.00**(polfit(np.log10(self.energy)))

        
        ########
        # Q_p_2s
        ########  
        coefs_qp2s = [-1.33340389e+00,  1.81849443e+00, -3.20438221e+00,  8.73242213e+00,
                 -8.72014392e+00,  3.83963229e+00, -8.43942678e-01,  8.84132749e-02,
                 -3.29638626e-03]
        polfit = Poly(coefs_qp2s)

        Q_p_2s = 10.00**(polfit(np.log10(self.energy)))

        ########
        # Q_p_2p
        ########  
        coefs_qp2p = [ 3.53592513e-01,  1.01935273e+00, -5.00234216e+00,  1.03067093e+01,
                  -9.73283938e+00,  4.61092269e+00, -1.20373119e+00,  1.67119555e-01,
                  -9.67776975e-03]
        polfit = Poly(coefs_qp2p)

        Q_p_2p = 10.00**(polfit(np.log10(self.energy)))

        ########
        # Q_p3
        ########          
        coefs_qp3s = [-1.32954426e+00,  3.24641453e+00, -7.92551893e+00,  1.29838478e+01,
                 -1.00084043e+01,  3.65608690e+00, -6.39881984e-01,  4.31933952e-02,
                  6.15522961e-05]
        polfit = Poly(coefs_qp3s)

        Q_p3 = 10.00**(polfit(np.log10(self.energy)))

        ########
        # Q_p_3s
        ########  
        coefs_qp3s = [-2.69721960e+00,  4.50212633e+00, -2.35886088e+01,  5.14218349e+01,
                 -4.86592762e+01,  2.37268261e+01, -6.34063602e+00,  8.86287912e-01,
                 -5.08139865e-02]
        polfit = Poly(coefs_qp3s)

        Q_p_3s = 10.00**(polfit(np.log10(self.energy)))

        ########
        # Q_p_3p
        ########  
        coefs_qp3p = [-1.59664156e+00,  2.62533789e+00, -7.72743744e+00,  1.58848880e+01,
                 -1.51939117e+01,  7.35580352e+00, -1.95091889e+00,  2.72419499e-01,
                 -1.57163068e-02]
        polfit = Poly(coefs_qp3p)

        Q_p_3p = 10.00**(polfit(np.log10(self.energy)))

        ########
        # Q_p_3d
        ########  
        coefs_qp3d = [-1.66661245e+00,  1.20858918e+00,  4.85307359e+00, -1.06404583e+01,
                  8.64651507e+00, -3.83846950e+00,  9.33555588e-01, -1.14318171e-01,
                  5.37941440e-03]
        polfit = Poly(coefs_qp3d)

        Q_p_3d = 10.00**(polfit(np.log10(self.energy)))

        #######
        # Q_1pP
        #######
        coefs_q1pP_1 = np.array((-1.55346521,  1.3669883 ,  1.85671678, -5.69724539,  9.64059189,
                                 -8.1251122 ,  3.44611815, -0.71654207,  0.05840993))
        coefs_q1pP_2 = np.array((2.76008009, -0.82241237))
        polfit_1 = Poly(coefs_q1pP_1)
        polfit_2 = Poly(coefs_q1pP_2)
        if np.max(self.energy > 1500):
            eind1 = np.where(self.energy <= 1500)[0][-1]
            Q_1pP_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_1pP_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_1pP = np.concatenate([Q_1pP_tmp1,Q_1pP_tmp2])
        else: 
            Q_1pP = 10**(polfit_1(np.log10(self.energy)))


        ######H
        # Q_1pH
        #######
        coefs_q1pH = np.array(( -0.02645452,   1.52461782,  -4.62208279,  10.30861507,
                                -10.41036241,   5.36557513,  -1.49664154,   0.21595013,
                                -0.01266363))
        polfit = Poly(coefs_q1pH)
        Q_1pH = 10**(polfit(np.log10(self.energy)))

        
        ######H
        # Q_1pE **
        #######
        eth = 13.6
        A = 0.18450
        B = np.array((-0.032226, -0.0343539, 1.4003, -2.8115, 2.2986))
        Q_1pE = A*np.log((self.energy*1e3)/eth)
        for i in range(len(B)):
            Q_1pE+= B[i]*(1 - eth/(self.energy*1e3))**i
        Q_1pE = Q_1pE*1e-13/(eth*(self.energy*1e3))*1e17

        #######
        # Q_1pE (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space
        coefs_q1pE_alt_1 = np.array((0.05984789, -0.80896196, -0.03479957, -0.21542295, -0.1196005 ,
                                 0.49888993,  0.17969381, -0.25437738, -0.12362548))
        coefs_q1pE_alt_2 = np.array((0.06223376, -0.87223181))
        polfit_1 = Poly(coefs_q1pE_alt_1)
        polfit_2 = Poly(coefs_q1pE_alt_2)
        if np.max(self.energy > 10):
            eind1 = np.where(self.energy <= 10)[0][-1]
            Q_1pE_alt_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_1pE_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_1pE_alt = np.concatenate([Q_1pE_alt_tmp1,Q_1pE_alt_tmp2])
        else: 
            Q_1pE_alt = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_12P **
        #######
        amu = 1.00797
        A = np.array((34.433, 44.507, 0.56870, 8.5476, 7.8501, 
                      -9.2217, 1.8020e-2, 1.6931, 1.9422e-3, 2.9067))
        val1 = (np.exp(-A[1]/(self.energy/amu))*np.log(1+A[2]*(self.energy/amu)))/(self.energy/amu)
        val2 = (A[3]*np.exp(-1.0*A[4]*(self.energy/amu)))/((self.energy/amu)**A[5])
        val3 = (A[6]*np.exp(-1.0*A[7]/(self.energy/amu)))/(1+A[8]*((self.energy/amu)**A[9]))
        Q_12P = 1e-16*A[0]*(val1+val2+val3)*1e17

        #######
        # Q_12P (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space. NOT AS GOOD AS THE IAEA FN!
        coefs_q12P_alt_1 = np.array((0.33301122,  1.66080004, -3.942614  ,  2.03507255,  2.46147057,
                                     -3.13088162,  1.34078766, -0.2581672 ,  0.01892247))
        coefs_q12P_alt_2 = np.array((2.74816921, -0.81042653))
        polfit_1 = Poly(coefs_q12P_alt_1)
        polfit_2 = Poly(coefs_q12P_alt_2)
        if np.max(self.energy > 5e3):
            eind1 = np.where(self.energy <= 5e3)[0][-1]
            Q_12P_alt_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_12P_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_12P_alt = np.concatenate([Q_12P_alt_tmp1,Q_12P_alt_tmp2])
        else: 
            Q_12P_alt = 10**(polfit_1(np.log10(self.energy)))


        #######
        # Q_12H
        #######
        coefs_q12H_1 = np.array(( 0.12915411,   1.59679531,   2.19750157, -10.63751475,
                                15.01683396, -11.36793345,   4.78531695,  -1.01708757,
                                0.0803237))
        coefs_q12H_2 = np.array((1.35631047, -0.6056955 ))
        polfit_1 = Poly(coefs_q12H_1)
        polfit_2 = Poly(coefs_q12H_2)
        if np.max(self.energy > 100):
            eind1 = np.where(self.energy <= 100)[0][-1]
            Q_12H_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_12H_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_12H = np.concatenate([Q_12H_tmp1,Q_12H_tmp2])
        else: 
            Q_12H = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_12H (2s)
        #######
        coefs_q12H_2s_1 = np.array(( 8.20641844,  -84.91941698,  345.48291415, -748.8303636 ,
                                    955.54571263, -738.38153022,  339.03405937,  -85.04061647,
                                    8.97014743))
        coefs_q12H_2s_2 = np.array((1.2894321 , -0.88626221))
        polfit_1 = Poly(coefs_q12H_2s_1)
        polfit_2 = Poly(coefs_q12H_2s_2)
        if np.max(self.energy > 100):
            eind1 = np.where(self.energy <= 100)[0][-1]
            Q_12H_2s_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_12H_2s_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_12H_2s = np.concatenate([Q_12H_2s_tmp1,Q_12H_2s_tmp2])
        else: 
            Q_12H_2s = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_12H (2p)
        #######
        coefs_q12H_2p_1 = np.array(( -0.12726034,   2.24142994,   3.93644845, -21.16601212,
                                     35.80462233, -33.48175971,  18.04849347,  -5.19161721,
                                     0.6141481))
        coefs_q12H_2p_2 = np.array((1.03055114, -0.50060726))
        polfit_1 = Poly(coefs_q12H_2p_1)
        polfit_2 = Poly(coefs_q12H_2p_2)
        if np.max(self.energy > 100):
            eind1 = np.where(self.energy <= 100)[0][-1]
            Q_12H_2p_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_12H_2p_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_12H_2p = np.concatenate([Q_12H_2p_tmp1,Q_12H_2p_tmp2])
        else: 
            Q_12H_2p = 10**(polfit_1(np.log10(self.energy)))

        ####### 
        # Q_12E **
        #######
        ## The below is only for E > 12.23 eV (which for us is fine)
        eth = 10.2
        A = np.array((1.4182, -20.877, 49.735, -46.249, 17.442))
        B = 4.4979
        Q_12E = 0
        for i in range(len(A)):
            Q_12E+= A[i]/((self.energy*1e3/eth)**i)  
        Q_12E+= B*np.log(self.energy*1e3/eth)
        Q_12E = Q_12E*5.984e-16/(eth*(self.energy*1e3/eth))*1e17

        #######
        # Q_12E (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space. NOT AS GOOD AS THE IAEA FN!
        coefs_q12E_alt_1 = np.array((0.11648916, -0.78548145, -0.06587174,  0.03026489, -0.00756062,
                                     0.00889251, -0.01263208,  0.00140759,  0.00236978))
        coefs_q12E_alt_2 = np.array((0.12140192, -0.82993883))
        polfit_1 = Poly(coefs_q12E_alt_1)
        polfit_2 = Poly(coefs_q12E_alt_2)
        if np.max(self.energy > 10):
            eind1 = np.where(self.energy <= 10)[0][-1]
            Q_12E_alt_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_12E_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_12E_alt = np.concatenate([Q_12E_alt_tmp1,Q_12E_alt_tmp2])
        else: 
            Q_12E_alt = 10**(polfit_1(np.log10(self.energy)))
       

        #######
        # Q_13P **
        #######
        amu = 1.00
        E = self.energy/amu
        A = np.array((6.1950, 35.773, 0.54818, 5.5162e-3,
                      0.29114, -4.5264, 6.0311, -2.0679))
        val1 = (np.exp(-1*A[1]/E) * np.log(1+A[2]*E))/E
        val2 = (A[3]*np.exp(-1*A[4]*E))/(E**A[5] + A[6]*E**A[7])
        Q_13P = 1e-16*A[0]*(val1+val2)*1e17

        #######
        # Q_13P (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space. NOT AS GOOD AS THE IAEA FN!
        coefs_q13P_alt_1 = np.array((-1.44140176,  2.32011149, -1.46732661, -1.07863213,  3.64643815,
                                     -3.1516196 ,  1.24799851, -0.23640629,  0.01739111))
        coefs_q13P_alt_2 = np.array((1.98572466, -0.80690771))
        polfit_1 = Poly(coefs_q13P_alt_1)
        polfit_2 = Poly(coefs_q13P_alt_2)
        if np.max(self.energy > 5e3):
            eind1 = np.where(self.energy <= 5e3)[0][-1]
            Q_13P_alt_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_13P_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_13P_alt = np.concatenate([Q_13P_alt_tmp1,Q_13P_alt_tmp2])
        else: 
            Q_13P_alt = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_13H
        #######
        coefs_q13H_1 = np.array(( -0.59153045,  3.09401076, -2.40878482,  0.74041026, -2.21063588,
                                  2.8966595 , -1.53333232,  0.36734892, -0.03333414))
        coefs_q13H_2 = np.array((1.49527064, -0.92320445))
        polfit_1 = Poly(coefs_q13H_1)
        polfit_2 = Poly(coefs_q13H_2)
        if np.max(self.energy > 1024):
            eind1 = np.where(self.energy <= 1024)[0][-1]
            Q_13H_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_13H_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_13H = np.concatenate([Q_13H_tmp1,Q_13H_tmp2])
        else: 
            Q_13H = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_13H (3s)
        #######
        coefs_q13H_3s_1 = np.array(( -0.74968012,  2.61016056, -1.16771074, -3.28147593,  3.51890403,
                                     -1.18827311,  0.01370219,  0.0688001 , -0.01017639))
        coefs_q13H_3s_2 = np.array((0.87560595, -0.97693724))
        polfit_1 = Poly(coefs_q13H_3s_1)
        polfit_2 = Poly(coefs_q13H_3s_2)
        if np.max(self.energy > 1024):
            eind1 = np.where(self.energy <= 1024)[0][-1]
            Q_13H_3s_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_13H_3s_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_13H_3s = np.concatenate([Q_13H_3s_tmp1,Q_13H_3s_tmp2])
        else: 
            Q_13H_3s = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_13H (3p)
        #######
        coefs_q13H_3p_1 = np.array(( -1.11958417,  3.77242633, -2.98073464,  1.27876324, -2.88232733,
                                     3.42697843, -1.75265739,  0.41218072, -0.03693575))
        coefs_q13H_3p_2 = np.array((1.3347965 , -0.90787652))
        polfit_1 = Poly(coefs_q13H_3p_1)
        polfit_2 = Poly(coefs_q13H_3p_2)
        if np.max(self.energy > 1024):
            eind1 = np.where(self.energy <= 1024)[0][-1]
            Q_13H_3p_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_13H_3p_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_13H_3p = np.concatenate([Q_13H_3p_tmp1,Q_13H_3p_tmp2])
        else: 
            Q_13H_3p = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_13H (3d)
        #######
        coefs_q13H_3d_1 = np.array(( -2.56733381,  4.53626236, -3.16124202,  1.33531155, -3.79221614,
                                     4.59968224, -2.3623394 ,  0.55769093, -0.05017271))
        coefs_q13H_3d_2 = np.array((0.42432161, -0.95933182))
        polfit_1 = Poly(coefs_q13H_3d_1)
        polfit_2 = Poly(coefs_q13H_3d_2)
        if np.max(self.energy > 1024):
            eind1 = np.where(self.energy <= 1024)[0][-1]
            Q_13H_3d_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_13H_3d_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_13H_3d = np.concatenate([Q_13H_3d_tmp1,Q_13H_3d_tmp2])
        else: 
            Q_13H_3d = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_13E **
        #######
        eth = 12.09
        A = np.array((0.42956, -0.58288, 1.0693, 0.0))
        B = 0.75488
        C = 0.38277
        Q_13E = 0
        for i in range(len(A)):
            Q_13E+= A[i]/((self.energy*1e3/eth)**i)
        Q_13E+= B*np.log(self.energy*1e3/eth)
        Q_13E = Q_13E * ((self.energy*1e3 - eth)/(self.energy*1e3))**C
        Q_13E = Q_13E * 5.984e-16/(eth*(self.energy*1e3/eth))*1e17

        #######
        # Q_13E (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space. NOT AS GOOD AS THE IAEA FN!
        coefs_q13E_alt_1 = np.array((-0.647381  , -0.709998  , -0.06548951, -0.59686341, -0.04866394,
                                     1.06982083,  0.27535206, -0.53390381, -0.22963316))
        coefs_q13E_alt_2 = np.array((-0.6484916 , -0.8291187))
        polfit_1 = Poly(coefs_q13E_alt_1)
        polfit_2 = Poly(coefs_q13E_alt_2)
        if np.max(self.energy > 10):
            eind1 = np.where(self.energy <= 10)[0][-1]
            Q_13E_alt_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_13E_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_13E_alt = np.concatenate([Q_13E_alt_tmp1,Q_13E_alt_tmp2])
        else: 
            Q_13E_alt = 10**(polfit_1(np.log10(self.energy)))


        #######
        # Q_2pP **
        #######
        amu = 1
        A = np.array((107.63, 29.860, 1.0176e6, 6.9713e-3, 
                      2.8488e-2, -1.8000, 4.7852e-2, -0.20923))
        val1 = (np.exp(-1*A[1]/(self.energy/amu)) * np.log(1+A[2]*(self.energy/amu)))/(self.energy/amu)
        val2 = (A[3]*np.exp(-1*A[4]*(self.energy/amu)))/((self.energy/amu)**A[5] + A[6]*(self.energy/amu)**A[7])
        Q_2pP = 1e-16*A[0]*(val1+val2)*1e17

        #######
        # Q_2pP (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space. NOT AS GOOD AS THE IAEA FN!
        coefs_q2pP_alt_1 = np.array(( 0.85250157,  1.52397353, -0.29357849,  1.39390691, -1.81142698,
                                      0.75437045, -0.09870323, -0.00771987,  0.00200768))
        coefs_q2pP_alt_2 = np.array((4.07769286, -0.917189 ))
        polfit_1 = Poly(coefs_q2pP_alt_1)
        polfit_2 = Poly(coefs_q2pP_alt_2)
        if np.max(self.energy > 5e3):
            eind1 = np.where(self.energy <= 5e3)[0][-1]
            Q_2pP_alt_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_2pP_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_2pP_alt = np.concatenate([Q_2pP_alt_tmp1,Q_2pP_alt_tmp2])
        else: 
            Q_2pP_alt = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_2pH
        #######

        #######
        # Q_2pE **
        #######
        eth = 3.4
        A = 0.14784
        B = np.array((0.0080871, -0.062270, 1.9414, -2.1980, 0.9584))
        Q_2pE = A*np.log((self.energy*1e3)/eth)
        for i in range(len(B)):
            Q_2pE+= B[i]*(1 - eth/(self.energy*1e3))**i
        Q_2pE = Q_2pE*1e-13/(eth*(self.energy*1e3))*1e17

        #######
        # Q_2pE (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space. NOT AS GOOD AS THE IAEA FN!
        coefs_q2pE_alt_1 = np.array((0.64219178, -0.89715704, -0.0933835 , -0.03788044,  0.17886587,
                                     0.14572807, -0.08295148, -0.10081622, -0.02376949))
        coefs_q2pE_alt_2 = np.array((0.63663501, -0.90108125))
        polfit_1 = Poly(coefs_q2pE_alt_1)
        polfit_2 = Poly(coefs_q2pE_alt_2)
        if np.max(self.energy > 10):
            eind1 = np.where(self.energy <= 10)[0][-1]
            Q_2pE_alt_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_2pE_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_2pE_alt = np.concatenate([Q_2pE_alt_tmp1,Q_2pE_alt_tmp2])
        else: 
            Q_2pE_alt = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_23P **
        #######
        amu = 1.
        A = np.array((394.51, 21.606, 0.62426, 0.013597, 0.16565, -0.8949))
        E = self.energy/amu
        val1 = (np.exp(-1*A[1]/E) * np.log(1+A[2]*E))/E
        val2 = (A[3]*np.exp(-1.0*A[4]*E))/((E)**A[5])
        Q_23P = 1e-16*A[0]*(val1+val2)*1e17

        #######
        # Q_23P (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space. NOT AS GOOD AS THE IAEA FN!
        coefs_q23P_alt_1 = np.array(( 1.65926194e+00,  6.04835441e-01, -1.77045318e-01,  6.51016242e-01,
                                      -8.08174811e-01,  3.61131822e-01, -7.04541087e-02,  4.86364776e-03,
                                       5.64902679e-05))
        coefs_q23P_alt_2 = np.array((3.90949306, -0.8397776))
        polfit_1 = Poly(coefs_q23P_alt_1)
        polfit_2 = Poly(coefs_q23P_alt_2)
        if np.max(self.energy > 5e3):
            eind1 = np.where(self.energy <= 5e3)[0][-1]
            Q_23P_alt_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_23P_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_23P_alt = np.concatenate([Q_23P_alt_tmp1,Q_23P_alt_tmp2])
        else: 
            Q_23P_alt = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_23H
        #######

        #######
        # Q_23E **
        #######
        eth = 1.899
        A = np.array((5.2373, 119.25, -595.39, 816.71))
        B = 38.906
        C = 1.3196
        Q_23E = 0
        for i in range(len(A)):
            Q_23E+= A[i]/((self.energy*1e3/eth)**(i))
        Q_23E+= B*np.log(self.energy*1e3/eth)
        Q_23E = Q_23E * ((self.energy*1e3 - eth)/(self.energy*1e3))**C
        Q_23E = Q_23E * 5.984e-16/(eth*(self.energy*1e3/eth))*1e17

        #######
        # Q_23E (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space. NOT AS GOOD AS THE IAEA FN!
        coefs_q23E_alt_1 = np.array((1.17602722, -0.84727972, -0.06286682,  0.05693193,  0.05395523,
                                     -0.03832005, -0.0331941 ,  0.00152561,  0.00235228))
        coefs_q23E_alt_2 = np.array((1.17839148, -0.86544671))
        polfit_1 = Poly(coefs_q23E_alt_1)
        polfit_2 = Poly(coefs_q23E_alt_2)
        if np.max(self.energy > 10):
            eind1 = np.where(self.energy <= 10)[0][-1]
            Q_23E_alt_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_23E_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_23E_alt = np.concatenate([Q_23E_alt_tmp1,Q_23E_alt_tmp2])
        else: 
            Q_23E_alt = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_3pP **
        #######
        amu = 1.0
        A = np.array((326.26, 13.608, 4.9910e3, 3.0560e-1, 
                      6.4364e-2, -0.14924, 3.1525, -1.6314))
        val1 = (np.exp(-1*A[1]/(self.energy/amu))*np.log(1+A[2]*(self.energy/amu)))/(self.energy/amu)
        val2 = (A[3]*np.exp(-1*A[4]*(self.energy/amu)))/((self.energy/amu)**A[5]+A[6]*(self.energy/amu)**A[7])
        Q_3pP = 1e-16*A[0]*(val1 + val2)*1e17

        #######
        # Q_3pP (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space. NOT AS GOOD AS THE IAEA FN!
        coefs_q3pP_alt_1 = np.array(( 2.36387575,  1.04585178, -0.14980269,  1.46309864, -3.12557449,
                                      2.26831053, -0.78795434,  0.13445398, -0.0090851 ))
        coefs_q3pP_alt_2 = np.array((4.46966962, -0.92106191))
        polfit_1 = Poly(coefs_q3pP_alt_1)
        polfit_2 = Poly(coefs_q3pP_alt_2)
        if np.max(self.energy > 5e3):
            eind1 = np.where(self.energy <= 5e3)[0][-1]
            Q_3pP_alt_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_3pP_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_3pP_alt = np.concatenate([Q_3pP_alt_tmp1,Q_3pP_alt_tmp2])
        else: 
            Q_3pP_alt = 10**(polfit_1(np.log10(self.energy)))

        #######
        # Q_3pH
        #######

        #######
        # Q_3pE **
        #######
        eth = 1.511
        A = 0.058463
        B = np.array((-0.051272, 0.85310, -0.57014, 0.76684, 0.0))
        Q_3pE = A*np.log((self.energy*1e3)/eth)
        for i in range(len(B)):
            Q_3pE+= B[i]*(1 - eth/(self.energy*1e3))**i
        Q_3pE = Q_3pE*1e-13/(eth*(self.energy*1e3))*1e17

        #######
        # Q_3pE (ALT)
        #######
        ## Obtained from fitting the IAEA Vol 4 data, and extending to high energy via a 
        ## linear fit in log-log space. NOT AS GOOD AS THE IAEA FN!
        coefs_q3pE_alt_1 = np.array((0.96598854, -0.96807841, -0.06514063,  0.02029611,  0.11111564,
                                     0.0366885 , -0.05579201, -0.03962078, -0.0074238))
        coefs_q3pE_alt_2 = np.array((0.95907927, -0.95892717))
        polfit_1 = Poly(coefs_q3pE_alt_1)
        polfit_2 = Poly(coefs_q3pE_alt_2)
        if np.max(self.energy > 10):
            eind1 = np.where(self.energy <= 10)[0][-1]
            Q_3pE_alt_tmp1 = 10**(polfit_1(np.log10(self.energy[:eind1+1])))
            Q_3pE_alt_tmp2 = 10**(polfit_2(np.log10(self.energy[eind1+1:])))
            Q_3pE_alt = np.concatenate([Q_3pE_alt_tmp1,Q_3pE_alt_tmp2])
        else: 
            Q_3pE_alt = 10**(polfit_1(np.log10(self.energy)))

        class cs_kerr_poly_out:
            def __init__(selfout):
                selfout.Q_p1 = Q_p1
                selfout.Q_p2 = Q_p2
                selfout.Q_p_2s = Q_p_2s
                selfout.Q_p_2p = Q_p_2p
                selfout.Q_p3 = Q_p3
                selfout.Q_p_3s = Q_p_3s
                selfout.Q_p_3p = Q_p_3p
                selfout.Q_p_3d = Q_p_3d
                selfout.Q_1pP = Q_1pP
                selfout.Q_1pH = Q_1pH
                selfout.Q_1pE = Q_1pE
                selfout.Q_1pE_alt = Q_1pE_alt 
                selfout.Q_12P = Q_12P
                selfout.Q_12P_alt = Q_12P_alt
                selfout.Q_12H = Q_12H
                selfout.Q_12H_2s = Q_12H_2s
                selfout.Q_12H_2p = Q_12H_2p
                selfout.Q_12E = Q_12E
                selfout.Q_12E_alt = Q_12E_alt
                selfout.Q_13P = Q_13P
                selfout.Q_13P_alt = Q_13P_alt
                selfout.Q_13H = Q_13H
                selfout.Q_13H_3s = Q_13H_3s
                selfout.Q_13H_3p = Q_13H_3p
                selfout.Q_13H_3d = Q_13H_3d
                selfout.Q_13E = Q_13E
                selfout.Q_13E_alt = Q_13E_alt
                selfout.Q_2pP = Q_2pP
                selfout.Q_2pP_alt = Q_2pP_alt
                # selfout.Q_2pH = Q_2pH
                selfout.Q_2pE = Q_2pE
                selfout.Q_2pE_alt = Q_2pE_alt
                selfout.Q_23P = Q_23P
                selfout.Q_23P_alt = Q_23P_alt
                # selfout.Q_23H = Q_23H
                selfout.Q_23E = Q_23E
                selfout.Q_23E_alt = Q_23E_alt
                selfout.Q_3pP = Q_3pP
                selfout.Q_3pP_alt = Q_3pP_alt
                # selfout.Q_3pH = Q_3pH
                selfout.Q_3pE = Q_3pE
                selfout.Q_3pE_alt = Q_3pE_alt
                selfout.energy = self.energy
                selfout.Units = 'energy in [keV], Q in [10^-17 cm^-2]'

        out = cs_kerr_poly_out()

        return out

################################################################################

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

        Their fits were in log space.

        Something weird is going on with the Q_1pE cross sections. Their
        fit is consistent with Canfield and Chang data points in their Table. 
        However, it lies well above my fits using IAEA. I think they have made 
        mistakes with eV vs keV???

        Graham Kerr
        July 2021

        '''

        ########################################################################
        # Go through each energy and calculate the cross section
        ########################################################################

        # for ind in range(self.nE):

        ########
        # Q_p1
        ########
        coefs_qp1 = np.array((-13.69, -2.03, 1.39, -.827, 0.988))
        polfit = Poly(coefs_qp1)
        Q_p1 = 10**(polfit(np.log10(self.energy)))
        

        ########
        # Q_p2
        ########          
        coefs_qp2 = np.array((-19.02, 5.59, -2.70, -0.00586, 0.0400))
        polfit = Poly(coefs_qp2)
        Q_p2 = 10**(polfit(np.log10(self.energy)))
       
        ########
        # Q_1pP
        ########
        coefs_q1pP = np.array((-18.17, 4.11, -2.11, 0.356, -0.0183))
        polfit = Poly(coefs_q1pP)
        Q_1pP = 10**(polfit(np.log10(self.energy)))

        ########
        # Q_1pH
        ########
        coefs_q1pH = np.array((-18.00, 2.81, -1.41, 0.265, -0.0228))
        polfit = Poly(coefs_q1pH)
        Q_1pH = 10**(polfit(np.log10(self.energy)))

        ########
        # Q_1pE
        ########
        coefs_q1pE = np.array((-26.97, 13.22, -5.18, 0.638))
        polfit = Poly(coefs_q1pE)
        Q_1pE = 10**(polfit(np.log10(self.energy)))
        einds = np.where(self.energy < 25.00)[0]
        Q_1pE[einds] = 0.0
         
        class cs_bw99_out:
            def __init__(selfout):
                selfout.Q_p1 = Q_p1/1e-17
                selfout.Q_p2 = Q_p2/1e-17
                selfout.Q_1pP = Q_1pP/1e-17
                selfout.Q_1pH = Q_1pH/1e-17
                selfout.Q_1pE = Q_1pE/1e-17
                selfout.energy = self.energy
                selfout.Units = 'energy in [keV], Q in [10^-17 cm^-2]'

        out = cs_bw99_out()

        return out

################################################################################

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

        NOTES
        ______
        
        A number of their fits go negative at certain points as the energy ranges
        fall outside of the ranges in the underlying data. This is a problem as 
        a number of the cross sections either dont work at all (the electron impacts)
        or go negative at ~100 keV, which isn't very high. I think that these values
        should not be used. 


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
            if self.energy[ind]*1e3 <= 63.0:
                a0,a1,a2,a3,a4 = -14.326, 1.6028, -4.9415e-2, 7.0030e-4, -3.7820e-6
                Q_1pE[ind] = a0 + (a1*(self.energy[ind]*1e3)**1) + (a2*(self.energy[ind]*1e3)**2) + (a3*(self.energy[ind]*1e3)**3) + (a4*(self.energy[ind]*1e3)**4)
            elif self.energy[ind]*1e3 > 63.0:
                a0,a1,a2,a3 = 7.4762, -0.02284, 3.0692e-5, -1.4225e-8
                Q_1pE[ind] = a0 + (a1*(self.energy[ind]*1e3)**1) + (a2*(self.energy[ind]*1e3)**2) + (a3*(self.energy[ind]*1e3)**3)

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
            Q_12E[ind] = a0 + (a1*(self.energy[ind]*1e3)**1) + (a2*(self.energy[ind]*1e3)**2) + (a3*(self.energy[ind]*1e3)**3) 

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
            if self.energy[ind]*1e3 <= 30.0:
                a0,a1,a2,a3 = -7.3427, 1.2622, -5.6240e-2, 7.8900e-4
                Q_13E[ind] = a0 + (a1*(self.energy[ind]*1e3)**1) + (a2*(self.energy[ind]*1e3)**2) + (a3*(self.energy[ind]*1e3)**3) 
            elif self.energy[ind]*1e3 > 30.0:
                a0,a1,a2,a3 = 0.89060, 0.002260, -3.4400e-4, 1.4179e-6
                Q_13E[ind] = a0 + (a1*(self.energy[ind]*1e3)**1) + (a2*(self.energy[ind]*1e3)**2) + (a3*(self.energy[ind]*1e3)**3)

            ########
            # Q_23E
            ########
            a0,a1,a2,a3 =  267.13, -3.2192, 1.2188e-2, 6.3314e-6
            Q_23E[ind] = a0 + (a1*(self.energy[ind]*1e3)**1) + (a2*(self.energy[ind]*1e3)**2) + (a3*(self.energy[ind]*1e3)**3) 
            
        
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

################################################################################
################################################################################
################################################################################

class cs_iaea93_Q_1pE:
    """
    This class holds the energy and cross sections for various processes, 
    from the XXXXXXXX

    Q_1pE H(n=1)* + e --> p* + e + e [E = 1.5e1 -- 1e4 eV]


    """

    def __init__(self):
        self.energy = np.array((1.5e1, 2e1, 4e1, 6e1, 8e1, 1e2, 2e2, 4e2, 
                                6e2, 8e2, 1e3, 2e3, 4e3, 6e3, 8e3, 1e4))/1e3
        self.Q_1pE = np.array((7.7e-18, 2.96e-17, 5.88e-17, 6.19e-17, 5.88e-17, 
                               5.51e-17, 3.95e-17, 2.41e-17, 1.75e-17, 1.39e-17, 
                               1.15e-17, 6.26e-18, 3.51e-18, 2.43e-18, 1.88e-18,
                               1.53e-18))/1e-17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'


################################################################################
################################################################################
################################################################################

class cs_iaea93_Q_12P:
    """
    This class holds the energy and cross sections for various processes, 
    from the XXXXXXXX

    Q_3pP p + H(n=1)* --> p + H(n=2)* + e [E = 6e2 -- 5e6 eV/amu]


    """
    def __init__(self):
        self.amu = 1.00797
        self.energy = np.array((6e2, 1e3, 2e3, 5e3,
                                1e4, 2e4, 5e4, 1e5,
                                2e5, 5e5, 1e6, 2e6, 
                                5e6))/1e3*self.amu
        self.Q_12P = np.array((5.77e-18, 2.34e-17, 3.04e-17, 3.63e-17,
                               2.80e-17, 5.10e-17, 1.00e-16, 9.00e-17,
                               6.38e-17, 3.55e-17, 2.09e-17, 1.20e-17,
                               5.50e-18))/1e-17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

################################################################################
################################################################################
################################################################################

class cs_iaea93_Q_12E:
    """
    This class holds the energy and cross sections for various processes, 
    from the XXXXXXXX

    Q_1pE H(n=1)* + e --> H(n=2)* + e [E = 1.02e1 -- 1e4 eV]


    """

    def __init__(self):
        self.energy = np.array((1.02e1, 2.00e1, 4.00e1, 6.00e1, 8.00e1, 
                                1.00e2, 2.00e2, 4.00e2, 6.00e2, 8.00e2,
                                1.00e3, 2.00e3, 4.00e3, 6.00e3, 8.00e3,
                                1.00e4))/1e3
        self.Q_12E = np.array((2.55e-17, 5.33e-17, 7.12e-17, 7.01e-17,
                               6.51e-17, 6.00e-17, 4.15e-17, 2.61e-17,
                               1.94e-17, 1.55e-17, 1.31e-17, 7.50e-18,
                               4.22e-18, 3.00e-18, 2.35e-18, 1.94e-18))/1e-17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'


################################################################################
################################################################################
################################################################################

class cs_iaea93_Q_13P:
    """
    This class holds the energy and cross sections for various processes, 
    from the XXXXXXXX

    Q_3pP p + H(n=1)* --> p + H(n=3)* + e [E = 5e2 -- 5e6 eV/amu]


    """
    def __init__(self):
        self.amu = 1.0
        self.energy = np.array((5e2, 1e3, 2e3, 5e3,
                                1e4, 2e4, 5e4, 1e5,
                                2e5, 5e5, 1e6, 2e6, 
                                5e6))/1e3*self.amu
        self.Q_13P = np.array((6.24e-20, 3.63e-19, 1.30e-18, 3.75e-18,
                               7.12e-18, 1.30e-17, 2.10e-17, 1.76e-17, 
                               1.18e-17, 6.32e-18, 3.73e-18, 2.13e-18,
                               9.89e-19))/1e-17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

################################################################################
################################################################################
################################################################################

class cs_iaea93_Q_13E:
    """
    This class holds the energy and cross sections for various processes, 
    from the XXXXXXXX

    Q_1pE H(n=1)* + e --> H(n=3)* + e [E = 1.24e1 -- 1e4 eV]


    """

    def __init__(self):
        self.energy = np.array((1.24e1, 2.00e1, 4.00e1, 6.00e1, 8.00e1, 
                                1.00e2, 2.00e2, 4.00e2, 6.00e2, 8.00e2,
                                1.00e3, 2.00e3, 4.00e3, 6.00e3, 8.00e3,
                                1.00e4))/1e3
        self.Q_13E = np.array((1.08e-18, 1.79e-17, 1.58e-17, 1.43e-17,
                               1.27e-17, 1.13e-17, 7.42e-18, 4.55e-18,
                               3.34e-18, 2.67e-18, 2.23e-18, 1.27e-18, 
                               7.18e-19, 5.12e-19, 3.98e-19, 3.31e-19))/1e-17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'



################################################################################
################################################################################
################################################################################

class cs_iaea93_Q_23P:
    """
    This class holds the energy and cross sections for various processes, 
    from the XXXXXXXX

    Q_3pP p + H(n=2)* --> p + H(n=3)* + e [E = 5e2 -- 5e6 eV/amu]


    """
    def __init__(self):
        self.amu = 1.
        self.energy = np.array((5e2, 1e3, 2e3, 5e3,
                                1e4, 2e4, 5e4, 1e5,
                                2e5, 5e5, 1e6, 2e6, 
                                5e6))/1e3*self.amu
        self.Q_23P = np.array((2.73e-16, 4.54e-16, 6.95e-16, 1.18e-15, 
                              1.68e-15, 2.03e-15, 1.77e-15, 1.33e-15,
                              8.59e-16, 4.33e-16, 2.49e-16, 1.39e-16,
                              6.28e-17))/1e-17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

################################################################################
################################################################################
################################################################################

class cs_iaea93_Q_23E:
    """
    This class holds the energy and cross sections for various processes, 
    from the XXXXXXXX

    Q_23E H(n=1)* + e --> H(n=3)* + e [E = 2.0e0 -- 1e4 eV]


    """
    def __init__(self):
        self.energy = np.array((2.00e0, 4.00e0, 6.00e0, 8.00e0,
                                1.00e1, 2.00e1, 4.00e1, 6.00e1, 8.00e1,
                                1.00e2, 2.00e2, 4.00e2, 6.00e2, 8.00e2,
                                1.00e3, 2.00e3, 4.00e3, 6.00e3, 8.00e3,
                                1.00e4))/1e3
        self.Q_23E = np.array((2.73e-15, 2.91e-15, 3.03e-15, 3.40e-15, 
                               3.57e-15, 2.87e-15, 1.78e-15, 1.34e-15, 1.10e-15,
                               9.32e-16, 5.45e-16, 3.17e-16, 2.28e-16, 1.80e-16,
                               1.50e-16, 8.32e-17, 4.55e-17, 3.21e-17, 2.50e-17,
                               2.04e-17))/1e-17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

################################################################################
################################################################################
################################################################################

class cs_iaea93_Q_2pP:
    """
    This class holds the energy and cross sections for various processes, 
    from the XXXXXXXX

    Q_3pP p + H(n=2)* --> p + p* + e [E = 5e2 -- 5e6 eV/amu]


    """
    def __init__(self):
        self.amu = 1.0
        self.energy = np.array((5e2, 1e3, 2e3, 5e3,
                                1e4, 2e4, 5e4, 1e5,
                                2e5, 5e5, 1e6, 2e6, 
                                5e6))/1e3*self.amu
        self.Q_2pP = np.array((2.06e-17, 6.97e-17, 2.13e-16, 8.20e-16,
                               2.03e-15, 3.54e-15, 2.82e-15, 1.71e-15,
                               9.35e-16, 3.99e-16, 2.13e-16, 1.12e-16,
                               4.84e-17))/1e-17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

################################################################################
################################################################################
################################################################################

class cs_iaea93_Q_2pE:
    """
    This class holds the energy and cross sections for various processes, 
    from the XXXXXXXX

    Q_2pE H(n=2)* + e --> p* + e + e [E = 4 -- 1e4 eV]


    """

    def __init__(self):
        self.energy = np.array((4.0e0, 6.0e0, 8.0e0, 1.0e1, 2.0e1, 4.0e1,
                                6.0e1, 8.0e1, 1.0e2, 2.0e2, 4.0e2, 6.0e2,
                                8.0e2, 1.0e3, 2.0e3, 4.0e3, 6.0e3, 8.0e3, 1e4))/1e3
        self.Q_2pE = np.array((2.19e-16, 8.51e-16, 1.08e-15, 1.16e-15, 9.92e-16, 6.49e-16, 
                               4.83e-16, 3.84e-16, 3.20e-16, 1.80e-16, 1.0e-16, 6.89e-17,
                               5.28e-17, 4.33e-17, 2.32e-17, 1.24e-17, 8.64e-18, 6.67e-18, 
                               5.42e-18))/1e-17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'


################################################################################
################################################################################
################################################################################

class cs_iaea93_Q_3pP:
    """
    This class holds the energy and cross sections for various processes, 
    from the XXXXXXXX

    Q_3pP p + H(n=3)* --> p + p* + e [E = 5e2 -- 5e6 eV/amu]


    """
    
    def __init__(self):
        self.amu = 1.0
        self.energy = np.array((5e2, 1e3, 2e3, 5e3,
                                1e4, 2e4, 5e4, 1e5,
                                2e5, 5e5, 1e6, 2e6, 
                                5e6))/1e3*self.amu
        self.Q_3pP = np.array((9.21e-16, 2.29e-15, 4.95e-15, 1.13e-14,
                               1.67e-14, 1.38e-14, 7.24e-15, 3.93e-15,
                               2.14e-15, 9.49e-16, 5.15e-16, 2.73e-16,
                               1.14e-16))/1e-17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

################################################################################
################################################################################
################################################################################

class cs_iaea93_Q_3pE:
    """
    This class holds the energy and cross sections for various processes, 
    from the XXXXXXXX

    Q_3pE H(n=3)* + e --> p* + e + e [E = 2 -- 1e4 eV]


    """

    def __init__(self):
        self.energy = np.array((2.0e0, 4.0e0, 6.0e0, 8.0e0, 1.0e1, 
                                2.0e1, 4.0e1, 6.0e1, 8.0e1, 1.0e2, 
                                2.0e2, 4.0e2, 6.0e2, 8.0e2, 1.0e3, 
                                2.0e3, 4.0e3, 6.0e3, 8.0e3, 1e4))/1e3
        self.Q_3pE = np.array((1.63e-15, 5.51e-15, 5.75e-15, 5.33e-15, 
                               4.84e-15, 3.12e-15, 1.79e-15, 1.26e-15,
                               9.72e-16, 7.93e-16, 4.17e-16, 2.17e-16, 
                               1.58e-16, 1.12e-16, 9.09e-17, 4.69e-17,
                               2.41e-17, 1.63e-17, 1.24e-17, 1.00e-17))/1e-17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

################################################################################
################################################################################
################################################################################

class cs_cariatore21:
    """
    This class holds the energy and cross sections for ionisation by H impact, 
    from Cariatore & Schultz 2021, ApJS 252. 
    https://ui.adsabs.harvard.edu/abs/2021ApJS..252....7C/abstract

    H* + H --> p* + H + e

    Q_1pH

    energy range = 1.36e-2 - 1e4 keV

    Note that they used a combination of theory, data, and their own calculations
    to create a 'recommended' curve

    """

    def __init__(self):
        self.energy = np.array((1.36e-2, 2e-2, 3e-2, 5e-2, 7e-2, 
                                1e-1, 2e-1, 3e-1, 5e-1, 7e-1, 
                                1e0, 2e0, 3e0, 5e0, 7e0, 
                                1e1, 1.2e1, 1.4e1, 1.6e1, 2e1, 
                                3e1, 5e1, 7e1, 
                                1e2, 2e2, 3e2, 5e2, 7e2, 
                                1e3, 2e3, 3e3, 5e3, 7e3, 1e4))
        self.Q_1pH = np.array((1e-99, 1e-24, 8e-22, 2e-20, 7e-20, 2.6e-19, 8.68e-19,
                               1.52e-18, 3.327e-18, 6.70e-18, 9.39e-18, 
                               1.67e-17, 2.25e-17, 3.62e-17, 4.60e-17, 6.89e-17, 
                               8.42e-17, 9.32e-17, 9.94e-17, 1.05e-16, 9.72e-17, 
                               8.35e-17, 7.14e-17, 5.70e-17, 3.58e-17, 2.53e-17, 
                               1.76e-17, 1.36e-17, 1.03e-17, 5.82e-18, 4.09e-18, 2.57e-18, 
                               1.88e-18, 1.35e-18))/1e-17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'


################################################################################
################################################################################
################################################################################

class cs_cheshire70:

    '''
    This class holds the energy and cross sections for charge transfer 
    from Cheshire et al 1970 J. Phys. B, 3 813, Table 5.
    https://ui.adsabs.harvard.edu/abs/1970JPhB....3..813C/abstract

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

################################################################################
################################################################################
################################################################################

class cs_ludde82:

    '''
    This class holds the energy and cross sections for charge transfer 
    from Ludde et al 1982 J. Phys. B, 15 2703, Table 1.
    https://ui.adsabs.harvard.edu/abs/1982JPhB...15.2703L/abstract

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


################################################################################
################################################################################
################################################################################

class cs_mb83:

    '''
    This class holds the energy and cross sections for hydrogen impact excitation
    to n = 2, from B M McLaughlin and K L Bell 1983 J. Phys. B: Atom. Mol. Phys. 16 3797,
    https://iopscience.iop.org/article/10.1088/0022-3700/16/20/016/pdf 
    
    Cross sections are of collisions between hydrogen and neutral H to 
    
    H* + H --> H(2s,2p)* + H 

    2s : Q_12H_2s
    2p : Q_12H_2p
    total : Q_12H (2s+2p)
 
    Energy range 1-100 keV

    '''

    def __init__(self):
        self.energy = np.array((1, 2.25, 4, 6.25, 9,
                                16, 25, 36, 49, 64, 81, 100))
        self.Q_12H_2s = np.array((1.52e-1, 4.37e-1, 4.53e-1, 3.58e-1,
                                  2.72e-1, 1.67e-1, 1.13e-1, 8.12e-2, 
                                  6.19e-2, 4.89e-2, 3.97e-2, 3.28e-2))/1e16*1e17
        self.Q_12H_2p = np.array((7.46e-2, 4.23e-1, 6.41e-1, 6.17e-1, 
                                  5.11e-1, 3.28e-1, 2.26e-1, 1.79e-1, 
                                  1.52e-1, 1.34e-1, 1.19e-1, 1.07e-1))/1e16*1e17
        self.Q_12H = self.Q_12H_2s+self.Q_12H_2p
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

################################################################################
################################################################################
################################################################################

class cs_mb87:

    '''
    This class holds the energy and cross sections for hydrogen impact excitation
    to n = 3, from B M McLaughlin and K L Bell 1987 J. Phys. B: Atom. Mol. Phys. 20 L217
    https://iopscience.iop.org/article/10.1088/0022-3700/20/7/005/pdf    
   
    Cross sections are of collisions between hydrogen and neutral H to 
    
    H* + H --> H(3s, 3p, 3d)* + H 

    3s : Q_13H_3s
    3p : Q_13H_3p
    3d : Q_13H_3d
    total : Q_13H (3s+3p+3d)
 
    Energy range 1-1024 keV

    '''
    def __init__(self):
        self.energy = np.array((  1.0,  1.5,   2.25,  4.0,   6.25,   9.0,
                                 16.0,  25.0,  36.0,  49.0,  64.0,  81.0,
                                100.0, 121.0, 144.0, 169.0, 196.0, 225.0, 
                                256.0, 289.0, 324.0, 361.0, 400.0, 441.0, 
                                484.0, 529.0, 576.0, 625.0, 676.0, 729.0,
                                784.0, 841.0, 900.0, 961.0, 1024.0
                                ))
        self.Q_13H_3s = np.array((1.772, 4.626, 8.428, 11.096, 9.571, 7.408,
                                4.451, 2.092, 2.046, 1.532, 1.198, 0.966,
                                0.796, 0.668, 0.568, 0.489, 0.426, 0.374, 
                                0.331, 0.295, 0.264, 0.238, 0.216, 0.196, 
                                0.179, 0.164, 0.151, 0.140, 0.129, 0.120, 
                                0.112, 0.104, 0.098, 0.091, 0.086))*0.1

        self.Q_13H_3p = np.array((0.759, 2.866, 7.363, 14.739, 16.153, 14.102, 
                                9.134, 6.153, 4.630, 3.819, 3.311, 2.928, 
                                2.606, 2.326, 2.078, 1.861, 1.670, 1.504, 
                                1.358, 1.231, 1.119, 1.021, 0.934, 0.857, 
                                0.789, 0.729, 0.674, 0.676, 0.583, 0.543, 
                                0.507, 0.475, 0.446, 0.419, 0.394))*0.1

        self.Q_13H_3d = np.array((0.027, 0.139, 0.450, 1.194, 1.467, 1.338,
                                  0.871, 0.585, 0.446, 0.375, 0.329, 0.291, 
                                  0.259, 0.229, 0.203, 0.179, 0.159, 0.142, 
                                  0.127, 0.114, 0.103, 0.093, 0.085, 0.077, 
                                  0.071, 0.065, 0.060, 0.055, 0.051, 0.048, 
                                  0.045, 0.042, 0.039, 0.036, 0.034))*0.1
   
        self.Q_13H = self.Q_13H_3s + self.Q_13H_3p + self.Q_13H_3d
   
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'


################################################################################
################################################################################
################################################################################

class cs_hill79:

    '''
    This class holds the energy and cross sections for hydrogen impact excitation
    to 2s, fromJ Hill et al 1979 J. Phys. B: Atom. Mol. Phys. 12 2875,
    https://iopscience.iop.org/article/10.1088/0022-3700/12/17/016/pdf
    
    Cross sections are of collisions between hydrogen and neutral H to 
    
    H* + H --> H(2s)* + H 

    2s : Q_12H_2s
    total : Q_12H (2s+2p)
 
    Energy range 1-25 keV

    '''

    def __init__(self):
        self.energy = np.array((2, 3, 4, 5, 6, 7, 8, 9, 
                                10, 11, 12, 13, 14, 15, 16, 17, 
                                18, 19, 20, 21, 22, 23, 24, 25
                                ))
        self.Q_12H_2s = np.array((0.6, 0.65, 0.77, 0.83, 0.94, 1.01,
                                  0.97, 1.10, 1.18, 1.19, 1.21, 1.32, 
                                  1.33, 1.33, 1.30, 1.32, 1.28, 1.26, 
                                  1.25, 1.11, 1.09, 1.11, 1.04, 1.00))
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

                                  

################################################################################
################################################################################
################################################################################


class cs_shakeshaft78:

    '''
    This class holds the energy and cross sections for charge transfer 
    from Shakeshaft 1978 Phys. Rev. A, 18, Table 2
    https://ui.adsabs.harvard.edu/abs/1978PhRvA..18.1930S/abstract
   
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


################################################################################
################################################################################
################################################################################


class cs_bates53:

    '''
    This class holds the energy and cross sections for charge transfer 
    from Bates & Dalgarno 1953 Proc. Phys. Soc. 66, Table 1
    https://ui.adsabs.harvard.edu/abs/1953PPSA...66..972B/abstract

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

################################################################################
################################################################################
################################################################################

class cs_winter09:

    '''
    This class holds the energy and cross sections for charge transfer 
    from Winter 2009 Phys. Rev. A. 80, Table 5. Essentially an update 
    to Shakeshaft 1978.
    https://ui.adsabs.harvard.edu/abs/2009PhRvA..80c2701W/abstract

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

################################################################################
################################################################################
################################################################################

class cs_belkic92:

    '''
    This class holds the energy and cross sections for charge transfer 
    from Belkic et al 1992  
    https://ui.adsabs.harvard.edu/abs/1992ADNDT..51...59B/abstract

    Cross sections are of collisions between protons and neutral H to 

    1s : Q_p1
    2s : Q_p_2s
    2p : Q_p_2p
    n=2: sum of 2s and 2p states, Q_p2
    3s : Q_p_3s
    3p : Q_p_3p
    3d : Q_p_3d
    n=3 : sum of 3s, 3p, 3d states, Q_p3
 
    Energy range  40 -- 1000 keV

    '''
    def __init__(self):
        amu = 1.00
        self.energy = np.array((40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 125.0, 150.0, 
                                200.0, 300.0, 400, 500.0, 600.0, 700., 800.0, 900.0, 1000.0, 
                                2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0))/amu
        self.Q_p1 = np.array((1.37e-16, 6.95e-17, 3.87e-17, 2.30e-17, 1.45e-17, 9.45e-18, 6.39e-18, 2.70e-18, 1.29e-18,
                              3.75e-19, 5.89e-20, 1.47e-20, 4.83e-21, 1.90e-21, 8.55e-22, 4.24e-22, 2.27e-22, 1.29e-22,
                              2.92e-24, 3.02e-25, 5.96e-26, 1.68e-26, 5.94e-27, 2.46e-27, 1.15e-27, 5.84e-28, 3.19e-28))*1e17
        self.Q_p_2s = np.array((2.81e-17, 1.44e-17, 8.00e-18, 4.71e-18, 2.91e-18, 1.87e-18, 1.24e-18, 5.05e-19, 2.33e-19,
                                6.43e-20, 9.38e-21, 2.24e-21, 7.13e-22, 2.75e-22, 1.22e-22, 5.96e-23, 3.16e-23, 1.79e-23,
                                3.89e-25, 3.98e-26, 7.80e-27, 2.19e-27, 7.73e-28, 3.20e-28, 1.49e-28, 7.59e-29, 4.14e-29))*1e17
        self.Q_p_2p = np.array((1.12e-17, 5.70e-18, 3.08e-18, 1.75e-18, 1.05e-18, 6.48e-19, 4.15e-19, 1.54e-19, 6.51e-20,
                                1.55e-20, 1.80e-21, 3.63e-22, 1.01e-22, 3.53e-23, 1.43e-23, 6.52e-24, 3.25e-24, 1.74e-24,
                                2.85e-26, 2.63e-27, 4.94e-28, 1.37e-28, 4.82e-29, 2.00e-29, 9.40e-30, 4.83e-30, 2.67e-30))*1e17
        self.Q_p2 = np.array((3.94e-17, 2.01e-17, 1.11e-17, 6.46e-18, 3.96e-18, 2.52e-18, 1.66e-18, 6.59e-19, 2.98e-19,
                              7.98e-20, 1.12e-20, 2.60e-21, 8.14e-22, 3.10e-22, 1.36e-22, 6.61e-23, 3.49e-23, 1.96e-23,
                              4.17e-25, 4.24e-26, 8.29e-27, 2.33e-27, 8.21e-28, 3.40e-28, 1.58e-28, 8.07e-29, 4.41e-29))*1e17
        self.Q_p_3s = np.array((8.96e-18, 4.65e-18, 2.59e-18, 1.53e-18, 9.43e-19, 6.06e-19, 4.02e-19, 1.62e-19, 7.41e-20,
                                2.03e-20, 2.91e-21, 6.88e-22, 2.18e-22, 8.38e-23, 3.69e-23, 1.18e-23, 9.55e-24, 5.39e-24,
                                1.16e-25, 1.19e-26, 2.33e-27, 6.53e-28, 2.30e-28, 9.54e-29, 4.44e-29, 2.26e-29, 1.23e-29))*1e17
        self.Q_p_3p = np.array((3.85e-18, 2.01e-18, 1.11e-18, 6.39e-19, 3.84e-19, 2.39e-19, 1.54e-19, 5.72e-20, 2.43e-20,
                                5.49e-21, 6.72e-22, 1.36e-22, 3.80e-23, 1.32e-23, 5.38e-24, 2.46e-24, 1.23e-24, 6.60e-25,
                                1.11e-26, 1.04e-27, 1.97e-28, 5.50e-29, 1.95e-29, 8.23e-30, 3.83e-30, 1.97e-30, 1.09e-30))*1e17
        self.Q_p_3d = np.array((1.02e-18, 4.39e-19, 2.11e-19, 1.10e-19, 6.05e-20, 3.51e-20, 2.13e-20, 7.04e-21, 2.74e-21,
                                5.82e-22, 6.05e-23, 1.18e-23, 3.33e-24, 1.18e-24, 4.95e-25, 2.33e-25, 1.20e-25, 6.67e-26,
                                1.41e-27, 1.51e-28, 3.09e-29, 9.03e-30, 3.31e-30, 1.42e-30, 6.80e-31, 3.56e-31, 1.99e-31))*1e17
        self.Q_p3 = np.array((1.38e-17, 7.10e-18, 3.91e-18, 2.28e-18, 1.39e-18, 8.80e-19, 5.77e-19, 2.26e-19, 1.01e-19,
                              2.66e-20, 3.64e-21, 8.35e-22, 2.59e-22, 9.82e-23, 4.28e-23, 2.07e-23, 1.09e-23, 6.11e-24,
                              1.29e-25, 1.31e-26, 2.55e-27, 7.17e-28, 2.53e-28, 1.05e-28, 4.89e-29, 2.49e-29, 1.36e-29))*1e17
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

################################################################################
################################################################################
################################################################################

class cs_tselia12:
    '''
    This class holds the energy and cross sections for charge transfer 
    from Tseliakhovich et al 2012 MNRAS 422, Table 4. 
    https://ui.adsabs.harvard.edu/abs/2012MNRAS.422.2357T/abstract

    Cross sections are of collisions between protons and neutral H to 

    1s : Q_p1
    2s : Q_p_2s
    2p : Q_p_2p
    n=2: sum of 2s and 2p states, Q_p2
    3s : Q_p_3s
    3p : Q_p_3p
    3d : Q_p_3d
    n=3 : sum of 3s, 3p, 3d states, Q_p3
 
    Energy range  5 -- 80 keV

    '''
    def __init__(self):
        self.energy = np.array((5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 60, 80))
        self.Q_p1 = np.array((1092, 925, 795, 695, 593, 425, 309, 224, 120, 42, 17))*0.1
        self.Q_p_2s = np.array((5.8, 12, 18, 27.1, 32.8, 39, 38.6, 34.9, 22, 8.6, 3.5))*0.1
        self.Q_p_2p0 = np.array((2.1, 3.0, 4.9, 6.6, 7.8, 7.9, 6.8, 5.7, 3.6, 1.3, 0.49))*0.1
        self.Q_p_2p1 = np.array((11.5, 12.0, 13, 11.6, 9.8, 6.5, 4.4, 3.0, 1.5, 0.40, 0.15))*0.1
        self.Q_p2 = self.Q_p_2p1 + self.Q_p_2p0 + self.Q_p_2s
        self.Q_p_3s = np.array((0.25, 0.55, 1.5, 3.2, 4.8, 8.6, 8.9, 8.8, 6.5, 2.7, 1.1))*0.1
        self.Q_p_3p0 = np.array((0.35, 0.67, 0.90, 1.6, 2.1, 2.5, 2.4, 2.0, 1.3, 0.5, 0.2))*0.1
        self.Q_p_3p1 = np.array((0.60, 0.89, 1.5, 1.6, 1.6, 1.7, 1.0, 0.8, 0.4, 0.15, 0.05))*0.1
        self.Q_p_3d0 = np.array((0.2, 0.25, 0.3, 0.4, 0.38, 0.35, 0.28, 0.21, 0.1, 0.03, 0.01))*0.1
        self.Q_p_3d1 = np.array((1.0, 1.2, 1.1, 0.8, 0.52, 0.28, 0.12, 0.07, 0.03, 0.01, 0.004))*0.1
        self.Q_p_3d2 = np.array((0.008, 0.03, 0.05, 0.04, 0.03, 0.03, 0.015, 0.01, 0.005, 0.002, 5e-4))*0.1
        self.Q_p3 = self.Q_p_3s + self.Q_p_3p0 + self.Q_p_3p1 + self.Q_p_3d0 + self.Q_p_3d1 + self.Q_p_3d2
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'

################################################################################
################################################################################
################################################################################

class cs_shah98:

    '''
    This class holds the energy and cross sections for H(1s)+p -> p + p + e 
    from Shah et al 1998 J. Phys. At. Mol. Opt. Phys 31.
    https://ui.adsabs.harvard.edu/abs/1998JPhB...31L.757S/abstract

    Q_1pP 
 
    Energy range  1.25 -- 9 keV

    '''
    def __init__(self):
        self.energy = np.array((1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
        self.Q_1pP = np.array((0.39, 0.55, 0.75, 1.38, 1.68, 3.2, 4.7, 5.8, 9.6, 11.3, 15.5))*0.1
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'


################################################################################
################################################################################
################################################################################

class cs_shah87a:

    '''
    This class holds the energy and cross sections for H(1s)+p -> p + p + e 
    from Shah et al 1987a J. Phys. B: Atom. Mol. Phys 20.
    https://ui.adsabs.harvard.edu/abs/1987JPhB...20.2481S/abstract
    
    Q_1pP 
 
    Energy range  9 -- 75 keV

    '''
    def __init__(self):
        self.energy = np.array((9.4, 11.4, 13.4, 15.4, 18.4, 22.4, 26.4, 32.4, 38.4, 48.4, 60.0, 75.0))
        self.Q_1pP = np.array((0.162, 0.245, 0.331, 0.438, 0.569, 0.779, 1.053, 1.227, 1.440, 1.300, 1.383, 1.291))*10.0
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'
 

################################################################################
################################################################################
################################################################################

class cs_shah81:

    '''
    This class holds the energy and cross sections for H(1s)+p -> p + p + e 
    from Shah & Gilbody 1981 J. Phys. B: Atom. Mol. Phys 14.
    https://ui.adsabs.harvard.edu/abs/1981JPhB...14.2361S/abstract

    Q_1pP 
 
    Energy range  38 -- 1500 keV

    '''
    def __init__(self):
        self.energy = np.array(( 38,   45,   54,   67,   80,  100,  120,   
                                140,        160,        180,  200,  230,  
                                260,  300,  350,  400,  450,  500,  550,  600,
                                700,  800,  900, 1000, 1150, 1300,
                               1500))
        self.Q_1pP = np.array((13.69, 13.99, 14.04, 13.42, 12.80, 11.16, 10.09, 
                                8.98,         8.27,         7.60,  7.07,  6.25,
                                5.80,  5.31,  4.61,  4.10,  3.68,  3.38,  3.19,
                                3.04,  2.64,  2.38,  2.18,  1.97,  1.75,  
                                1.58,  1.38))
        self.units = 'energy in [keV], Q in [10^-17 cm^-2]'


################################################################################
################################################################################
################################################################################


def cs_polyfit(energy, csec, emin = -100.0, emax=-100.0, 
               order = 3, log10E=False, log10Q = False):
    """
    Performs a nth degree polynomial fit to the cross sections at certain energies
    
    INPUTS
    ______
    
    energy -- the projectile energy in keV
    csec  -- the cross sections in 10^-17 cm^2
    emin -- the minimum energy at which to perform the fit 
            [optional, defaults to energy[0]]
    emax -- the maximum energy at which to perform the fit 
            [optional, defaults to energy[-1]]
    order -- the degree of the fits 
              [optional, defaults to 3]
    log10E -- perform the fit in log10 E space
              [optional, defaults to False]
    log10Q -- perform the fit in log10 Q space
              [optional, defaults to True]

    OUTPUTS
    _______

    The fitting results

    NOTES
    _____

    Graham Kerr
    July 2021

    """
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



################################################################################
################################################################################
################################################################################


def cs_chebfit(energy, csec, emin = -100.0, emax = -100.0,
                    deg = 8):
    """
    Performs a fit to the cross sections at certain energies, using Chebyshev 
    polynomials (see methods below for the functional form)
    
    INPUTS
    ______
    
    energy -- the projectile energy in keV
    csec  -- the cross sections in 10^-17 cm^2
    emin -- the minimum energy at which to perform the fit 
            [optional, defaults to energy[0]]
    emax -- the maximum energy at which to perform the fit 
            [optional, defaults to energy[-1]]
    deg -- the degree of the fits 
           [optional, defaults to 8]

    OUTPUTS
    _______

    The fitting coeficients and covariance matrix

    NOTES
    _____

    The fits are done using the variable x (see below) and 
    cross sections in natural log space (ln Q). 

    Graham Kerr
    July 2021

    """
    
    if emax == -100.0:
        emax = energy[-1]
    if emin == -100.0:
        emin = energy[0]

    eind1 = np.where(energy >= emin)[0][0]
    eind2 = np.where(energy <= emax)[0][-1]+1

    energy = energy[eind1:eind2]

    x = ((np.log(energy)-np.log(emin)) - (np.log(emax) - np.log(energy)))/(np.log(emax) - np.log(emin))

    csec = csec[eind1:eind2]

    # y0_guess = csec[0] 

    # paramsguess = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    paramsguess = np.ones(deg+1)
    popt_cheb, pcov_cheb=curve_fit(chebyshev_tofit, 
                                   x, np.log(csec), 
                                   p0 = paramsguess)

    #yvals = OZpy.CrossSections.exponential_fn(xvals, 
                                  # *popt_exp)

    return popt_cheb, pcov_cheb

 
################################################################################
################################################################################
################################################################################

def chebyshev_fn(energy, coefs, deg, emin, emax):
    """
    Evaluates the Chebyshev function at energy E, given coeficients. 

    Y = A_0 / 2 + Sigma_j [A_j * T(x)_j]

    Where 
    x is a variable containing the energy 
          x = [{ln(energy) - ln(Emin)} - {ln(Emax) - ln(energy)}]/{ln(Emax) - ln(Emin)}
          for Emin and Emax the minimum and maximum energy bounds for which the coefs 
          are valid
    T(x)_j are the j-th Chebyshev polynomials
    A_j are coefficients 
  
    INPUTS
    _______

    energy -- the energy at which to evaluate the function, in keV
    coefs -- the coeficients of the function (presumably from some fit)
    deg -- the degree of the Chebyshev function 
    emin -- the minimum energy at which the function coefs are valid
    emax -- the maximum energy at which the function coefs are valid

    OUTPUTS
    _______

    chebfn -- the function evaluated at energies provided. These are, 
              in our case, ln(cross sections) for cross sections in 10^-17 cm^2

    NOTES
    ______

    To be used to evalutate the cross sections using the coeficients from fitting
    the data.

    Graham Kerr
    July 2021
    """

    ## Turn to np array if an integer or float are provided
    if type(energy) == int:
        energy = np.array(energy)
    if type(energy) == float:
        energy = np.array(energy)
    if type(energy) == tuple:
        energy = np.array(energy)

    nE = len(energy)
   
    ## The fitting variable
    x = ((np.log(energy)-np.log(emin)) - (np.log(emax) - np.log(energy)))/(np.log(emax) - np.log(emin))

    ## Cheb polynomials
    tfn = np.zeros([nE,9], dtype = np.float64)
    tfn[:,0] = x**0
    tfn[:,1] = x
    tfn[:,2] = 2*x**2 - 1
    tfn[:,3] = 4*x**3 - 3*x
    tfn[:,4] = 8*x**4 - 8*x**2 + 1
    tfn[:,5] = 16*x**5 - 20*x**3 + 5*x
    tfn[:,6] = 32*x**6 - 48*x**4 + 28*x**2 - 1
    tfn[:,7] = 64*x**7 - 112*x**5 + 56*x**3 - 7*x
    tfn[:,8] = 128*x**8 - 256*x**6 + 160*x**4 - 32*x**2 + 1

    chebfn = coefs[0]/2*tfn[:,0]

    for ind in range(1,deg):
        chebfn+=coefs[ind]*tfn[:,ind]

    return chebfn

################################################################################
################################################################################
################################################################################

def chebyshev_tofit(x, *coefs):
    """
    Fits a function of the form

    Y = A_0 / 2 + Sigma_j [A_j * T(x)_j]

    Where 
    x is a variable containing the energy 
    T(x)_j are the j-th Chebyshev polynomials
    A_j are coefficients

    This is largely used as the function in the fitting routine above. A seperate
    method can evaluate the Chebyshev fns given energy and coeficients. 

    INPUTS
    ______

    x -- the values at which to evaluate the function 
    coefs -- the coeficients of the function

    OUTPUTS
    _______

    Y - the function evaluated at X 

    NOTES
    _____


    Graham Kerr
    July 2021

    """
    
    ## Turn to np array if an integer or float are provided
    if type(x) == int:
        x = np.array(x)
    if type(x) == float:
        x = np.array(x)
    if type(x) == tuple:
        x = np.array(x)

    nE = len(x)
    
    ## Cheb polynomials
    tfn = np.zeros([nE,9], dtype = np.float64)
    tfn[:,0] = x**0
    tfn[:,1] = x
    tfn[:,2] = 2*x**2 - 1
    tfn[:,3] = 4*x**3 - 3*x
    tfn[:,4] = 8*x**4 - 8*x**2 + 1
    tfn[:,5] = 16*x**5 - 20*x**3 + 5*x
    tfn[:,6] = 32*x**6 - 48*x**4 + 28*x**2 - 1
    tfn[:,7] = 64*x**7 - 112*x**5 + 56*x**3 - 7*x
    tfn[:,8] = 128*x**8 - 256*x**6 + 160*x**4 - 32*x**2 + 1

    chebfn = np.zeros([nE], dtype = np.float64)
    
    chebfn = coefs[0]/2 * tfn[:,0] 
    for ind in range(1,len(coefs)):
        chebfn+= coefs[ind]*tfn[:,ind]
    

    return chebfn
  
################################################################################
################################################################################
################################################################################

################################################################################
################################################################################
################################################################################

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

################################################################################
################################################################################
################################################################################

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

################################################################################
################################################################################
################################################################################

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

################################################################################
################################################################################
################################################################################

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