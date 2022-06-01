import numpy as np 
import OrrallZirkerPy as OZpy
from OrrallZirkerPy.EnergyToVel import energy2vel 
from OrrallZirkerPy.AtomData import EinsteinA
import sys


"""
Solves for the number density of suprathermal He II in the excited state, 
following the arguments in Peter et al 1990. That is, the relative population 
of suprathermal He I, He II and He III are calculated by solving the statistical 
equilibrium eqns. The system of equations is constrained to keep the total equal 
to unity (i.e. fraction of He I + He II + He III = 1).

The assumption here is that the beam ions are initially He III (alphas), and that
over time various processes can convert the alphas to other charge states. 

Then, once those proportions are known the amount of He II in the excited level 
is calculated. 

Note that for most cases, especially when we are first using alpha beams, we likely
just take a proton beam simulation and suppose that some fraction of those protons 
are alphas (e.g. 5% according to Peter et al 1990).


Steps:

1) Read in atmosphere object
2) Read in active cross sections object 
3) Read in suprathermal proton distribution fn (e.g from RADYN+FP) 
 ... loop through energy
4) Sum up the contributions to each level 
5) Subtract the transitions out of each level
6) Set up the matrix equation 
7) Use lin. alg. solver to return the array of pops
8) Return the pops

Graham Kerr
May 2022

"""
def CalcPopsHe(csec, atmos, nthmp, isum = -1):

	########################################################################
    # Some preliminary set up
    ########################################################################
    
    ## Number of energy bins, and convert to velocity
    nE_cs = csec.nE
    energy = csec.energy
    vel_cs = energy2vel(energy,particle='alpha')
    nLev = csec.nLev
    
    
    ## Add additional dimnesions to atmosphere object if necessary 
    ## (taken care of in the main routine, but just in case)
    if len(atmos.nElec.shape) == 2:
        nDim1 = atmos.nElec.shape[0]
        nDim2 = atmos.nElec.shape[1]
        if (nDim1 != nthmp.fe.shape[0]) or (nDim2 != nthmp.fe.shape[1]):
        	sys.exit('>>> Exiting... \nDimenions of ambient particles dont match dimensions of injected proton spectrum.\nCheck your depth and time grid\n')
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
    		sys.exit('>>> Exiting... \nDimenions of ambient particles dont match dimensions of injected proton spectrum.\nCheck your depth and time grid\n')
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
        	sys.exit('>>> Exiting... \nDimenions of ambient particles dont match dimensions of injected proton spectrum.\nCheck your depth and time grid\n')
    if len(atmos.nElec.shape) != len(nthmp.fe.shape)-1:
        sys.exit('>>>Exiting... \n Dimenions of ambient particles dont match dimensions of injected proton spectrum.\nCheck your depth and time grid\n')
    


    ## Factor to convert cross sections to cm^2
    cs2cm = 1e-17


    ## Grab the Einstein Coefs
    Aij = EinsteinA(nLev = nLev, species = 'He')
    

    NPops = np.zeros([nDim1, nDim2, nLev+1, nE_cs], dtype = np.float64)
    NPops_HeIIex = np.zeros([nDim1, nDim2, nE_cs], dtype = np.float64)

    for eind in range(0,nE_cs):
    # for eind in range(100,101):
    
        ## Interpolate to energies at which the user has requested (energy_cs). This could
        ## be done in the main routine, but doesn't hurt doing it here, in case this is 
        ## run standalone. **** DO I NEED TO DELETE THIS COMMENT... ARE WE STILL INERPOLATING,
        ## OR DO WE JUST REQUEST THE SAME ENERGIES AS IN THE FP ARRAY???
        NthmProtons = nthmp.fe[:,:,eind]#*vel_cs[eind]
        
        ########################################################################
        # Turn the Cross Sections into Rates
        ########################################################################

        ###### extend the cross section arrays to cover the height or time arrays
        ###### as needed

        C_HeH = np.repeat(csec.cs_HeH[eind, np.newaxis], nDim1, axis=0)
        C_HeH = np.repeat(C_HeH[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_HeE = np.repeat(csec.cs_HeE[eind, np.newaxis], nDim1, axis=0)
        C_HeE = np.repeat(C_HeE[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_HeP = np.repeat(csec.cs_HeP[eind, np.newaxis], nDim1, axis=0)
        C_HeP = np.repeat(C_HeP[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_HeCT = np.repeat(csec.cs_HeCT[eind, np.newaxis], nDim1, axis=0)
        C_HeCT = np.repeat(C_HeCT[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_He2H = np.repeat(csec.cs_He2H[ eind, np.newaxis], nDim1, axis=0)
        C_He2H = np.repeat(C_He2H[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_He2E = np.repeat(csec.cs_He2E[eind, np.newaxis], nDim1, axis=0)
        C_He2E = np.repeat(C_He2E[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_He2P = np.repeat(csec.cs_He2P[ eind, np.newaxis], nDim1, axis=0)
        C_He2P = np.repeat(C_He2P[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_He2CT = np.repeat(csec.cs_He2CT[eind, np.newaxis], nDim1, axis=0)
        C_He2CT = np.repeat(C_He2CT[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_He2HCT = np.repeat(csec.cs_He2HCT[eind, np.newaxis], nDim1, axis=0)
        C_He2HCT = np.repeat(C_He2HCT[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_He3HCT = np.repeat(csec.cs_He3HCT[eind, np.newaxis], nDim1, axis=0)
        C_He3HCT = np.repeat(C_He3HCT[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_He3exH = np.repeat(csec.cs_He3exH[eind, np.newaxis], nDim1, axis=0)
        C_He3exH = np.repeat(C_He3exH[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_He2exH = np.repeat(csec.cs_He2exH[eind, np.newaxis], nDim1, axis=0)
        C_He2exH = np.repeat(C_He2exH[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_He2exE = np.repeat(csec.cs_He2exE[eind, np.newaxis], nDim1, axis=0)
        C_He2exE = np.repeat(C_He2exE[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        C_He2exP = np.repeat(csec.cs_He2exP[eind, np.newaxis], nDim1, axis=0)
        C_He2exP = np.repeat(C_He2exP[:, np.newaxis], nDim2, axis=1)*cs2cm*vel_cs[eind]

        ########################################################################
        # Create the rates matrix
        ########################################################################

        Pij = np.zeros([nDim1, nDim2, nLev+1, nLev+1], dtype = np.float64)

        ## We are going to construct the Helium rates matrix 'by hand', filling
        ## in each row. 

        ## P_11 
        Pij[:,:,0,0] = -1*atmos.nHyd*C_HeH + -1*atmos.nElec*C_HeE + -1*atmos.nProt*C_HeP + -1*atmos.nProt*C_HeCT
        ## P_12
        Pij[:,:,0,1] = atmos.nHyd*C_He2HCT
        ## P_13
        Pij[:,:,0,2] = 0.0
        ## P_21
        Pij[:,:,1,0] = atmos.nHyd*C_HeH + atmos.nElec*C_HeE + atmos.nProt*C_HeP + atmos.nProt*C_HeCT
        ## P_22
        Pij[:,:,1,1] = -1*atmos.nHyd*C_He2HCT + -1*atmos.nHyd*C_He2H + -1*atmos.nElec*C_He2E + -1*atmos.nProt*C_He2P + -1*atmos.nProt*C_He2CT 
        ## P_23
        Pij[:,:,1,2] = atmos.nHyd*C_He3HCT 
        ## P_31
        Pij[:,:,2,0] = 0.0
        ## P_32
        Pij[:,:,2,1] = atmos.nHyd*C_He2H + atmos.nElec*C_He2E + atmos.nProt*C_He2P + atmos.nProt*C_He2CT
        ## P_33
        Pij[:,:,2,2] = -1*atmos.nHyd*C_He3HCT


        ## Pij * Npop = X
        ## X is the result of each equation, which is zero since we bring creation 
        ## and destruction on same side. It is a [Nlev+1 x 1] array... but we also 
        ## want to store this for each height or time point, if they are defined. 
        ##
        ## For helium we are solving for each charge state: He I, He II, He III, 
        ## so two bound levels (nLev = 2) plus the continuum.
        X = np.zeros([nDim1, nDim2, nLev+1], dtype = np.float64)

        ## Replace one of the equations with the particle conservation equation
        Pij[:,:,isum,:] = 1.

        NthmProtons = 1.0 ## temporary... normalises to 1
        X[:,:,isum] = NthmProtons 

        ########################################################################
        # Solve the Statistical Equilibrium Equations
        ########################################################################

        NPops[:,:,:,eind] = np.linalg.solve(Pij, X) 

       	NPops_HeIIex[:,:,eind] = (atmos.nHyd*C_He3exH*NPops[:,:,2,eind] + atmos.nHyd*C_He2exH*NPops[:,:,1,eind] + atmos.nElec*C_He2exE*NPops[:,:,1,eind] + atmos.nProt*C_He2exP*NPops[:,:,1,eind])/Aij.Aij[0,0]	


    ## If necessary remove extraneous dimensions
    # Npop = np.squeeze(Npop)

    class SupraThermPops_out:
        def __init__(selfout):
            selfout.NPops = NPops
            selfout.NPops_HeIIex = NPops_HeIIex
            selfout.energy = energy
            selfout.nLev = nLev
            selfout.species = csec.species
            selfout.Units = 'energy in [keV], Pops in [particles cm^-3 keV^-1]'
                   
    return SupraThermPops_out()


