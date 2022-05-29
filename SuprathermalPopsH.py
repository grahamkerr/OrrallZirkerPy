import numpy as np 
import OrrallZirkerPy as OZpy
from OrrallZirkerPy.EnergyToVel import energy2vel 
from OrrallZirkerPy.AtomData import EinsteinA
import sys


"""
Solves the statistical equilibrium equations to obtain the number density of 
the J suprathermal states.

Eliminates isum = -1 (by default) level (the suprathermal proton density) 
equation and replaces it with particle conservation equation.


[Pij] x [N_pops] = [X]

X = zeroes if we collate creation and destruction within Rates matrix Pij

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

"""
def CalcPopsH(csec, atmos, nthmp, isum = -1):
    

    ########################################################################
    # Some preliminary set up
    ########################################################################
    
    ## Number of energy bins, and convert to velocity
    nE_cs = csec.nE
    energy = csec.energy
    vel_cs = energy2vel(energy)
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
    Aij = EinsteinA(nLev = nLev)


    NPops = np.zeros([nDim1, nDim2, nLev+1, nE_cs], dtype = np.float64)


    for eind in range(0,nE_cs):
    # for eind in range(100,101):
    
        ## Interpolate to energies at which the user has requested (energy_cs). This could
        ## be done in the main routine, but doesn't hurt doing it here, in case this is 
        ## run standalone. 
        NthmProtons = nthmp.fe[:,:,eind]#*vel_cs[eind]

        ########################################################################
        # Turn the Cross Sections into Rates
        ########################################################################

        ###### extend the cross section arrays to cover the height or time arrays
        ###### as needed

        ## Charge Exchange 
        C_ij_CX = np.repeat(csec.cs_CX[:, :, eind, np.newaxis], nDim1, axis=2)
        C_ij_CX = np.repeat(C_ij_CX[:, :, :, np.newaxis], nDim2, axis=3)

        ## Proton collisions
        C_ij_colP = np.repeat(csec.cs_colP[:, :, eind, np.newaxis], nDim1, axis=2)
        C_ij_colP = np.repeat(C_ij_colP[:, :, :, np.newaxis], nDim2, axis=3)

        ## Hydrogen collisions
        C_ij_colH = np.repeat(csec.cs_colH[:, :, eind, np.newaxis], nDim1, axis=2)
        C_ij_colH = np.repeat(C_ij_colH[:, :, :, np.newaxis], nDim2, axis=3)
 
        ## Electron collisions
        C_ij_colE = np.repeat(csec.cs_colE[:, :, eind, np.newaxis], nDim1, axis=2)
        C_ij_colE = np.repeat(C_ij_colE[:, :, :, np.newaxis], nDim2, axis=3)

  
        C_ij_CX = np.multiply(C_ij_CX*cs2cm, atmos.nHyd) * vel_cs[eind]
        C_ij_colP = np.multiply(C_ij_colP*cs2cm, atmos.nProt) * vel_cs[eind]
        C_ij_colH = np.multiply(C_ij_colH*cs2cm, atmos.nHyd) * vel_cs[eind]
        C_ij_colE = np.multiply(C_ij_colE*cs2cm, atmos.nElec) * vel_cs[eind]


        ########################################################################
        # Create the rates matrix
        ########################################################################

        # Pij = np.zeros([nLev+1, nLev+1, nDim1, nDim2], dtype = np.float64)
        # for iind in range(0,nLev+1):
        #     for jind in range(0,nLev+1):
        #         if iind!=jind:
        #             Pij[iind, jind, :, :] = Pij[iind, jind, :, :] + C_ij_colP[jind,iind, :, :] ## Creation
        #             Pij[iind, jind, :, :] = Pij[iind, jind, :, :] + C_ij_colH[jind,iind, :, :] ## Creation
        #             Pij[iind, jind, :, :] = Pij[iind, jind, :, :] + C_ij_colE[jind,iind, :, :] ## Creation
        #             Pij[iind, jind, :, :] = Pij[iind, jind, :, :] + C_ij_CX[jind,iind, :, :] ## Creation (via Charge Ex)
        #             Pij[iind, iind, :, :] = Pij[iind,iind, :, :] - (C_ij_colP[iind,jind, :, :] + 
        #                                                 C_ij_colH[iind,jind, :, :] +
        #                                                 C_ij_colE[iind,jind, :, :])
        #             Pij[iind, iind, :, :] = Pij[iind,iind, :, :] - Aij.Aij[iind,jind]
        ####### The above was an older version where I had height and time at the end for the Pij array. 
        ####### I rearranged (below) in order to be able to pass the whole lot to the linalg solver.

        Pij = np.zeros([nDim1, nDim2, nLev+1, nLev+1], dtype = np.float64)

        # print(Pij.shape)
        # print(C_ij_colP.shape)
        # print(nDim1)
        # print(nDim2)
        for iind in range(0,nLev+1):
            for jind in range(0,nLev+1):
                if iind!=jind:
                    Pij[:, :, iind, jind] = Pij[:, :, iind, jind] + C_ij_colP[jind,iind, :, :] ## Creation
                    Pij[:, :, iind, jind] = Pij[:, :, iind, jind] + C_ij_colH[jind,iind, :, :] ## Creation
                    Pij[:, :, iind, jind] = Pij[:, :, iind, jind] + C_ij_colE[jind,iind, :, :] ## Creation
                    Pij[:, :, iind, jind] = Pij[:, :, iind, jind] + C_ij_CX[jind,iind, :, :] ## Creation (via Charge Ex)

                    ## Destruction via collisions, then via emission 
                    Pij[:, :, iind, iind] = Pij[:, :, iind,iind] - (C_ij_colP[iind,jind, :, :] + 
                                                        C_ij_colH[iind,jind, :, :] +
                                                        C_ij_colE[iind,jind, :, :])
                    Pij[:, :, iind, iind] = Pij[:, :, iind,iind] - Aij.Aij[iind,jind]

        ## Pij * Npop = X
        ## X is the result of each equation, which is zero since we bring creation 
        ## and destruction on same side. It is a [Nlev+1 x 1] array... but we also 
        ## want to store this for each height or time point, if they are defined. 
        X = np.zeros([nDim1, nDim2, nLev+1], dtype = np.float64)

        ## Replace one of the equations with the particle conservation equation
        Pij[:,:,isum,:] = 1.0

        # NthmProtons = 1.0 ## temporary... normalises to 1, I guess?
        X[:,:,isum] = NthmProtons 

        ########################################################################
        # Solve the Statistical Equilibrium Equations
        ########################################################################

        NPops[:,:,:,eind] = np.linalg.solve(Pij, X)

    ## If necessary remove extraneous dimensions
    # Npop = np.squeeze(Npop)

    class SupraThermPops_out:
        def __init__(selfout):
            selfout.NPops = NPops
            selfout.energy = energy
            selfout.nLev = nLev
            selfout.species = csec.species
            selfout.Units = 'energy in [keV], Pops in [particles cm^-3 keV^-1]'
            # selfout.C_ij_colP = C_ij_colP
            # selfout.C_ij_colH = C_ij_colH
            # selfout.C_ij_colE = C_ij_colE
            # selfout.C_ij_CX = C_ij_CX
            # selfout.Pij = Pij
    # out = SupraThermPops_out()

       
    return SupraThermPops_out()

