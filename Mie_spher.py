# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:11:17 2021
    Refer to the website:
        https://www.giss.nasa.gov/staff/mmishchenko/t_matrix.html
    Refer to the Fortran code:
        spher.f
@author: Tianfeng-Pink
"""

'''
    Mie():
    Input parameters:                                                   
     NDISTR specifies the type of particle size distribution as follows:        
     NDISTR = 1 - modified gamma distribution                         
          [Eq. (5.242) of Ref. 1]                                        
              AA=alpha                                                
              BB=r_c                                                  
              GAM=gamma                                               
     NDISTR = 2 - log normal distribution                             
          [Eq. (5.243) of Ref. 1]                                        
              AA=r_g                                                  
              BB=[ln(sigma_g)]**2                                      
     NDISTR = 3 - power law distribution                              
          [Eq. (5.244) of Ref. 1]                                        
               AA=r_eff (effective radius)                            
               BB=v_eff (effective variance)                          
               Parameters R1 and R2 (see below) are calculated automatically for given AA and BB                      
     NDISTR = 4 - gamma distribution                                  
          [Eq. (5.245) of Ref. 1]                                        
               AA=a                                                   
               BB=b                                                   
     NDISTR = 5 - modified power law distribution                     
          [Eq. (5.246) of Ref. 1]                                        
               BB=alpha                                               
     NDISTR = 6 - bimodal volume log normal distribution              
              [Eq. (5.247) of Ref. 1]             
              AA1=r_g1                                                
              BB1=[ln(sigma_g1)]**2                                   
              AA2=r_g2                                                
              BB2=[ln(sigma_g2)]**2                                   
              GAM=gamma                                               

   - R1 and R2 - minimum and maximum radii in the size distribution for NDISTR=1-4 and 6.         
         R1 and R2 are calculated automatically for the power law distribution with given 
         r_eff and v_eff but must be specified for other distributions. For the modified 
         power law distribution (NDISTR=5), the minimum radius is 0, R2 is the maximum radius, 
         and R1 is the intermediate radius at which the n(r)=const dependence is replaced 
         by the power law dependence.                                                  

   - LAM - wavelength of the incident light in the surrounding medium

   - Important:  r_c, r_g, r_eff, a, LAM, R1, and R2                  
                 must be in the same units of length (e.g., microns)  

   - MRR and MRI - real and imaginary parts of the relative refractive index 
           ********************(MRI must be non-negative)*******************                        

   - N - number of integration subintervals on the interval (R1, R2) of particle radii                                            
   - NP - number of integration subintervals on the interval (0, R1) for the modified 
           power law distribution                      
   - NK - number of Gaussian division points on each of the integration subintervals                                    

   - NPNA - number of scattering angles at which the scattering matrix is computed                                            
        (see the PARAMETER statement in subroutine MATR).             
        The corresponding scattering angles are given by 180*(I-1)/(NPNA-1) (degrees), 
        where I numbers the angles.     
        This way of selecting scattering angles can be easily changed at the beginning of 
        subroutine MATR by properly modifying the following lines:                                          
                                                                      
          N=NPNA                                                       
          DN=1D0/DFLOAT(N-1)                                          
          DA=DACOS(-1D0)*DN                                           
          DB=180D0*DN                                                 
          TB=-DB                                                      
          TAA=-DA                                                     
          DO 500 I1=1,N                                               
             TAA=TAA+DA                                               
             TB=TB+DB                                                 
                                                                      
        and leaving the rest of the subroutine intact.                
        This flexibility is provided by the fact that after the expansion coefficients 
        AL1,...,BET2 (see below) are computed by subroutine SPHER, the scattering matrix       
        can be computed for any set of scattering angles without repeating Lorenz-Mie calculations.                                   

   - DDELT - desired numerical accuracy of computing the scattering matrix elements
    
     Output information:                                                 
                                                                      
   - REFF and VEFF - effective radius and effective variance of the size distribution 
       [Eqs. (5.248)-(5.250) of Ref. 2]   
   - CEXT and CSCA - average extinction and scattering cross sections per particle                                         
   - <COS> - asymmetry parameter                 
   - ALBEDO - single-scattering albedo                                
   - <G> - average projected area per particle                        
   - <V> - average volume per particle                                
   - Rvw - volume-weighted average radius                             
   - <R> - average radius                                                
   - F11 = a_1, F33 = a_3, F12 = b_1, and F34 = b_2 - elements of the 
          normalized scattering matrix given by Eq. (4.65) of Ref. 1.       
   - ALPHA1, ..., BETA2 - coefficients appearing in the expansions of the elements of 
          the normalized scattering matrix in generalized spherical functions 
          [Eqs. (4.75)-(4.80) of Ref. 1].              

   The first line of file 10 provides the single-scattering albedo and the number of 
       numerically significant expansion coeffieients ALPHA1, ..., BETA2 followed
       by the values of the expansion coefficients.
                                                                      
   Optical efficiency factors QEXT and QSCA are computed as QEXT=CEXT/<G> and QSCA=CSCA/<G>. 
   The absorption cross section is given by CABS=CEXT-CSCA. The absorption efficiency factor      
   is equal to QABS=QEXT-QSCA.                                        
                                                                       
   To calculate scattering and absorption by a monodisperse particle with radius r = AA, 
   use the following options:                                                                    
        BB=1D-1                                                       
        NDISTR=4                                                      
        NK=1                                                          
        N=1                                                           
        R1=AA*0.9999999 D0                                            
        R2=AA*1.0000001 D0
        
'''
    
import numpy as np
from tqdm import tqdm
import time

## TODO: Define constant
Nmie    = 10000
NPL     = 2*Nmie
NDRDI   = 3*Nmie
NPNA    = 19                                # Number of scattering angles at which the scattering matrix is computed
NGRAD   = 100000

# 20210423 debug
############################### CalScaMatrix ##################################
## TODO: Calculate the scattering matrix
def CalScaMatrix(alpha, beta, L_max):
    N   = NPNA
    DA  = np.pi/(N - 1.0)           # radian
    DB  = 180.0/(N - 1.0)           # degree
    ScaMat = np.zeros([N, 5])       # N*[ScaAng, F11, F12, F33, F34]
    ScaMat[:, 0] = np.arange(0, N*DB, DB)
    for i in range(N):
        U = np.cos(i*DA)
        F11, F2, F3, F12, F34 = 0.0, 0.0, 0.0, 0.0, 0.0
        P1, P2, P3, P4 = 0.0, 0.0, 0.0, 0.0
        PP1 = 1.0
        PP2 = 0.25*(1.0 + U)**2
        PP3 = 0.25*(1.0 - U)**2
        PP4 = np.sqrt(6.0)*0.5*(U**2 - 1.0)
        
        for L1 in range(L_max + 1):
            F11 += alpha[0, L1]*PP1
            if L1 != L_max:
                P = ((2*L1 + 1)*U*PP1 - L1*P1)/(L1 + 1.0)
                P1 = PP1
                PP1 = P
            if L1 >= 2:
                F2 += (alpha[1, L1] + alpha[2, L1])*PP2
                F3 += (alpha[1, L1] - alpha[2, L1])*PP3
                F12 += beta[0, L1]*PP4
                F34 += beta[1, L1]*PP4
                if L1 != L_max:
                    P = ((2*L1 + 1)*(L1*(L1 + 1)*U - 4.0)*PP2 - (L1 + 1)*(L1**2 - 4)*P2)/(L1*((L1 + 1.0)**2 - 4))
                    P2 = PP2
                    PP2 = P
                    P = ((2*L1 + 1)*(L1*(L1 + 1)*U + 4.0)*PP3 - (L1 + 1)*(L1**2 - 4)*P3)/(L1*((L1 + 1.0)**2 - 4))
                    P3 = PP3
                    PP3 = P
                    P = ((2*L1 + 1)*U*PP4 - np.sqrt(L1**2 - 4)*P4)/np.sqrt(np.sqrt((L1 + 1.0)**2 - 4))
                    P4 = PP4
                    PP4 = P
        F33 = (F2 - F3)*0.5
        ScaMat[i, 1:] = [F11, F12, F33, F34]
    
    return ScaMat       
############################### CalScaMatrix END ##############################
    
# 20210420 debug
################################## ANGL #######################################
## TODO: Calculation of the angular functions PI(N) and TAU(N) for given argument.
def Angl(Nmax, mu, coeff):
    '''
    mu = COS(THETA)
    
    '''
    Pin, Taun = np.zeros(Nmax), np.zeros(Nmax)
    P1, P2 = 0.0, 1.0
    for N in range(Nmax):
        S = mu*P2
        T = S - P1
        Taun[N] = (N + 1)*T - P1
        Pin[N] = P2
        P1 = P2
        P2 = S + coeff[N]*T
    # for N in range(1, Nmax + 1):
    #     S = mu*P2
    #     T = S - P1
    #     Taun[N - 1] = N*T - P1
    #     Pin[N - 1] = P2
    #     P1 = P2
    #     P2 = S + coeff[N - 1]*T
         
    return Pin, Taun
################################## ANGL END ###################################

# 20210421 debug
################################## GENER ######################################
## TODO: Calculation of the generalized spherical functions
def Gener(mu, L1_max, coef):
    '''
    P[0, 1] = 0.0
    P[1, 1] = 0.0
    P[0, 2] = 0.0
    P[1, 2] = 0.0
    P[0, 3] = 0.0
    P[1, 3] = 0.0
    '''
    L1_max = int(L1_max)
    P       = np.zeros([L1_max, 4])
    P[0, 0] = 1.0
    P[1, 0] = mu
    P[2, 0] = 0.5*(3.0*mu**2 - 1.0)
    P[2, 1] = 0.25*(1.0 + mu)**2 
    P[2, 2] = 0.25*(1.0 - mu)**2 
    P[2, 3] = 0.25*np.sqrt(6.0)*(mu**2 - 1.0)
    C11, C12 = coef[:-1, 0]*coef[:-1, 1]*mu, coef[:-1, 0]*np.arange(2, L1_max-1)
    C21, C22 = coef[:-1, 4]*(coef[:-1, 5]*mu-coef[:-1, 6]), coef[:-1, 4]*coef[:-1, 7]
    C31      = coef[:-1, 4]*(coef[:-1, 5]*mu+coef[:-1, 6])
    C41, C42 = coef[:-1, 2]*coef[:-1, 1]*mu, coef[:-1, 2]*coef[:-1, 3]
    for i in range(2, L1_max - 1):
        P[i+1, 0] = C11[i-2]*P[i, 0] - C12[i-2]*P[i-1, 0]
        P[i+1, 1] = C21[i-2]*P[i, 1] - C22[i-2]*P[i-1, 1]
        P[i+1, 2] = C31[i-2]*P[i, 2] - C22[i-2]*P[i-1, 2]
        P[i+1, 3] = C41[i-2]*P[i, 3] - C42[i-2]*P[i-1, 3]      
    return P
################################# GENER END ###################################

# 20210424 debug
################################## DISTRB #####################################
def Distrb(NNK, YY, WY, NDISTR, AA, BB, GAM, R_1, R_2, \
           AA1 = None, BB1 = None, AA2 = None, BB2 = None):
    
    assert NDISTR != 3, 'POWER LAW DISTRIBUTION HAS BEEN Prohibited)'
    if NDISTR == 2:
        print('LOG NORMAL DISTRIBUTION: r_g = ' + str(AA) + \
              ', [ln(sigma_g)]**2 = ' + str(BB))
        WY[:NNK] *= np.exp(-(np.log(YY[:NNK]/AA))**2 *0.5/BB)/YY[:NNK]
#    elif NDISTR == 3:
#        print('POWER LAW DISTRIBUTION OF HANSEN & TRAVIS 1974 (Prohibited)')
#        WY /= YY**3
    elif NDISTR == 4:
        print('GAMMA DISTRIBUTION, a = ' + str(AA) + ', b = ' + str(BB))
        WY[:NNK] *= (YY[:NNK]**((1.0 - 3.0*BB)/BB))*np.exp(-YY[:NNK]/(AA*BB))
        
    elif NDISTR == 5:
        print('MODIFIED POWER LAW DISTRIBUTION, ALPHA = ' + str(BB))
        ind     = WY[:NNK] > R_1
        WY[ind] *= (YY[ind]/R_1)**BB
        
    elif NDISTR == 6:
        print('BIMODAL VOLUME LOG NORMAL DISTRIBUTION')
        print('r_g1 = ' + str(AA1) + ', [ln(sigma_g1)]**2 = ' + str(BB1))
        print('r_g2 = ' + str(AA2) + ', [ln(sigma_g2)]**2 = ' + str(BB2))
        print('gamma = ' + str(GAM))
        WY[:NNK] *= (np.exp(-(np.log(YY[:NNK]/AA1))**2*0.5/BB1) + \
               GAM*np.exp(-(np.log(YY[:NNK]/AA2))**2*0.5/BB2))/YY[:NNK]**4
        
    else:
        print('MODIFIED GAMMA DISTRIBUTION, ALPHA = ' + str(AA) + \
              ', r_c = ' + str(BB) + ', GAMMA = ' + str(GAM))
        WY[:NNK] *= YY[:NNK]**AA*np.exp(-AA/GAM*((YY[:NNK]/BB)**GAM))
        
    WY      *= 1.0/np.sum(WY)        
    G       = np.sum(YY**2*WY)
    Reff    = np.sum(YY**3*WY)/G
    Veff    = np.sum((YY - Reff)**2 * YY**2 * WY)/(G*Reff**2)
    Rvw     = np.sum(YY**4*WY)/np.sum(YY**3*WY)
    Volume  = np.sum(YY**3*WY)*4.0*np.pi/3.0
    Rmean   = np.sum(YY*WY)
    area    = G*np.pi
    
    return WY, G, Reff, Veff, area, Volume, Rvw, Rmean                  
################################## DISTRB END #################################

# 20210424 debug
##################################### GAUSS ###################################
def Gauss(N, IND1, IND2):
    Z, W    = np.zeros(N), np.zeros(N)    
    K       = N//2 + N % 2
    for I in range(K):
        M = N - I
        if I == 0: X = 1.0 - 2.0/((N + 1.0)*N)         
        if I == 1: X = (Z[-1] - 1.0)*4.0 + Z[-1]  
        if I == 2: X = (Z[-2] - Z[-1])*1.6 + Z[-2]
        if I >  2: X = (Z[M] - Z[M + 1])*3.0 + Z[M + 2]
        if I == K - 1 and N % 2 == 1: X = 0.0
            
        NITER   = 0
        CHECK   = 1e-16
        PB      = 1.0
        NITER   += 1
        if NITER > 100:
            CHECK *= 10.0
        PC = X
        DJ = 1.0
        for J in range(1, N):
            DJ += 1.0
            PA = PB
            PB = PC
            PC = X*PB + (X*PB - PA)*(DJ - 1.0)/DJ
        PA = 1.0/((PB - X*PC)*N)
        PB = PA*PC*(1.0 - X**2)
        X  -= PB
        while np.abs(PB) > CHECK*np.abs(X):
            PB      = 1.0
            NITER   += 1
            if NITER > 100: CHECK *= 10.0
            PC = X
            DJ = 1.0
            for J in range(1, N):
                DJ += 1.0
                PA = PB
                PB = PC
                PC = X*PB + (X*PB - PA)*(DJ - 1.0)/DJ
            PA = 1.0/((PB - X*PC)*N)
            PB = PA*PC*(1.0 - X**2)
            X  -= PB
        Z[M - 1] = X
        W[M - 1] = PA**2*(1.0 - X**2)
        if IND1 == 0: W[M - 1] *= 2.0
        if I == K - 1 and N % 2 == 1: break
        Z[I] = -Z[M - 1]
        W[I] = W[M - 1]
    # if IND2 == 1:
    #     print('***  POINTS AND WEIGHTS OF GAUSSIAN QUADRATURE FORMULA OF' + \
    #           str(N) + '-TH ORDER ***')
    # for I in range(K):
    #     print('X(' + str(I + 1) + ') = ' + str(-Z[I]) + \
    #           ', W(' + str(I + 1) + ') = ' + str(W[I]))
    if IND1 != 0: Z = (1.0 + Z)*0.5
    return Z, W
################################ GAUSS END ####################################

##################################### SPHER #####################################    
def Spher(AA, BB, GAM, Lam, MRR, MRI, R_1, R_2, N, NP, NDISTR, NK, AA1, BB1, AA2, BB2, DDELT):
    '''
    N: the number of integration subintervals on the interval (R1, R2) of particle radii
    NK: the number of Gaussian division points on each of the integration subintervals
    *** MRI < 0 ***
    '''
    # 20210425 debug
    assert NDISTR != 3, 'POWER LAW DISTRIBUTION HAD BEEN Prohibited)'
#    if NDISTR == 3:
#        R_1, R_2 = Power(AA, BB, N)
    
    # 20210425 debug
## TODO: CALCULATION OF GAUSSIAN DIVISION POINTS AND WEIGHTS
    assert NK <= NPL, 'NK IS GREATER THAN NPL. EXECUTION TERMINATED'    
    X, W    = Gauss(NK, 0, 0)    
    NNPK    = int(NK**2) if NDISTR == 5 else 0
    NNK     = int(N*NK + NNPK)
    assert NNK <= NPL, 'NNK IS GREATER THAN NGRAD, EXECUTION TERMINATED'
    WN      = 2.0*np.pi/Lam
    RX      = R_2*WN
    M       = int(RX + 4.05*np.cbrt(RX) + 8.0) 
    assert M <= Nmie, 'TOO MANY MIE COEFFICIENTS. INCREASE NMIE.'
    
    MArr        = np.arange(M)
    coeff       = np.zeros([3, M])
    coeff[0, :] = (MArr + 2.0)/(MArr + 1.0)
    coeff[1, :] = 2.0*MArr + 3.0
    coeff[2, :] = 0.5*(2*MArr + 3.0)/((MArr + 1.0)*(MArr + 2.0))
    del MArr
    # for I in range(M):
    #     coeff[0, I] = (I + 2)/(I + 1)
    #     coeff[1, I] = (2.0*I + 3.0)
    #     coeff[2, I] = 0.5*(2.0*I + 3.0)/((I + 1)*(I + 2))
    
    NG      = 2*M - 1
    L1_max  = 2*M
    RMR     =  MRR/(MRR**2+MRI**2)
    RMI     = -MRI/(MRR**2+MRI**2) 
    
    # 20210425 debug
    YY, XX, WY = np.zeros(NGRAD), np.zeros(NPL), np.zeros(NGRAD)
    if NDISTR == 5:
        XX[:NK]    = R_1/NP*0.5*(X[:NK] + 1.0)
        YY[:NP*NK] = np.tile(XX[:NK], NP) + np.repeat(R_1/NP*np.arange(NP), NK)
        WY[:NP*NK] = np.tile(W[:NK]*R_1/NP*0.5, NP)
    XX[:NK]              = (R_2 - R_1)/N*0.5*(X[:NK] + 1.0)
    YY[NNPK:(N*NK+NNPK)] = np.tile(XX[:NK], N) + np.repeat((R_2 - R_1)/N*np.arange(N), NK) + R_1
    WY[NNPK:(N*NK+NNPK)] = np.tile(W[:NK]*(R_2 - R_1)/N*0.5, N)
    
    WY, G, Reff, Veff, area, Volume, Rvw, Rmean = Distrb(NNK, YY, WY, NDISTR, \
                                                         AA, BB, GAM, R_1, R_2, \
                                                         AA1, BB1, AA2, BB2)
    C_ext = 0.0
    C_sca = 0.0
    X, W = Gauss(NG, 0, 0)
    
    # 20210425 debug
## TODO: AVERAGING OVER SIZES
    F_11, F_33, F_12, F_34  = np.zeros(NG), np.zeros(NG), np.zeros(NG), np.zeros(NG)
    DR, DI, HI, RPSI        = np.zeros(NDRDI), np.zeros(NDRDI), np.zeros(NDRDI), np.zeros(NDRDI)
    PSI, AR, AI, BR, BI     = np.zeros(Nmie), np.zeros(Nmie), np.zeros(Nmie), np.zeros(Nmie), np.zeros(Nmie)
    for i in tqdm(range(NNK)):
    # for i in range(NNK):
        RXR     = MRR*YY[i]*WN
        RXI     = MRI*YY[i]*WN
        DC      = np.cos(YY[i]*WN)
        DS      = np.sin(YY[i]*WN)
        CXR     = RXR/(RXR**2+RXI**2)
        CXI     = -RXI/(RXR**2+RXI**2)

## TODO: CALCULATION OF THE MIE COEFFICIENTS
        M_1 = int(YY[i]*WN + 4.05*np.cbrt(YY[i]*WN) + 8)
        M_2 = M_1 + 2 + int(1.2*np.sqrt(YY[i]*WN)) + 5
        assert M_2 <= NDRDI, 'M2.GT.NDRDI. EXECUTION TERMINATED'
        Q_max = max(M_1, np.sqrt(RXR**2+RXI**2))
        M_4 = int(6.4*np.cbrt(Q_max) + Q_max) + 8
        assert M_4 <= NDRDI, 'M4.GT.NDRDI. EXECUTION TERMINATED'
        D_4 = M_4 + 1
        DR[M_4-1], DI[M_4-1] = D_4*CXR, D_4*CXI
        HI[0]   = DS + DC/(YY[i]*WN)
        HI[1]   = 3.0*HI[0]/(YY[i]*WN) - DC
        PSI[0]  = DS/(YY[i]*WN) - DC
        RPSI[M_2-1] = YY[i]*WN/(2*M_2 + 1)
        for j in range(1, M_2 - 1):
            J_1 = M_2 - j
            RPSI[J_1 - 1] = 1.0/((2*J_1 + 1.0)/(YY[i]*WN) - RPSI[J_1])
        
        for J in range(1, M_4):
            J1 = M_4-J+1
            OR = DR[J1-1] + J1*CXR
            OI = DI[J1-1] + J1*CXI
            DR[J1-2] = J1*CXR - OR/(OR**2+OI**2)
            DI[J1-2] = J1*CXI + OI/(OR**2+OI**2)
        del OR, OI, J1, J
        # for J in range(2, M_4 + 1):
        #     J1 = M_4-J+2
        #     J2 = J1-1
        #     DJ = J1
        #     FR = DJ*CXR
        #     FI = DJ*CXI
        #     OR = DR[J1-1] + FR
        #     OI = DI[J1-1] + FI
        #     ORI = 1.0/(OR**2+OI**2)
        #     DR[J2-1] = FR-OR*ORI
        #     DI[J2-1] = FI+OI*ORI
        
        # 20210420 debug
        for J in range(1, M_1 - 1):
            HI[J + 1]   = (2*(J + 1) + 1)*HI[J]/(YY[i]*WN) - HI[J - 1]
            PSI[J]      = RPSI[J]*PSI[J - 1]
        PSI[M_1 - 1] = RPSI[M_1 - 1]*PSI[M_1 - 2]
        
        # 20210427 debug
        OR  = DR[0]*RMR - DI[0]*RMI + 1.0/(YY[i]*WN)
        OR1 = OR*PSI[0] - DS
        OI  = DR[0]*RMI + DI[0]*RMR
        OI1 = OI*PSI[0]
        OR2 = OR*PSI[0] - OI*HI[0] - DS
        OI2 = OR*HI[0] + OI*PSI[0] - DC
        OAB = 1.0/(OR2**2 + OI2**2)
        AR[0] = (OR1*OR2 + OI1*OI2)*OAB
        AI[0] = (OR2*OI1 - OR1*OI2)*OAB
        OR  = DR[0]*MRR - DI[0]*MRI + 1.0/(YY[i]*WN)
        OR1 = OR*PSI[0] - DS
        OI  = DI[0]*MRR + DR[0]*MRI
        OI1 = OI*PSI[0]
        OR2 = OR*PSI[0] - OI*HI[0] - DS
        OI2 = OR*HI[0] + OI*PSI[0] - DC
        OAB = 1.0/(OR2**2 + OI2**2)
        BR[0] = (OR1*OR2 + OI1*OI2)*OAB
        BI[0] = (OR2*OI1 - OR1*OI2)*OAB
        del OR, OR1, OR2, OI, OI1, OI2, OAB
        
        # 20210420 debug
        OR      = DR[1:M_1]*RMR - DI[1:M_1] *RMI + np.arange(2, M_1 + 1)/(YY[i]*WN)
        OI      = DR[1:M_1]*RMI + DI[1:M_1] *RMR
        OR1     = OR*PSI[1:M_1] - PSI[:M_1 - 1]
        OR2     = OR*PSI[1:M_1] - OI*HI[1:M_1] - PSI[:M_1 - 1]
        OI1     = OI*PSI[1:M_1]
        OI2     = OR*HI[1:M_1] + OI*PSI[1:M_1] - HI[:M_1 - 1]
        AR[1:M_1]   = (OR1*OR2 + OI1*OI2)/(OR2**2 + OI2**2)
        AI[1:M_1]   = (OR2*OI1 - OR1*OI2)/(OR2**2 + OI2**2)
        OR      = DR[1:M_1]*MRR - DI[1:M_1] *MRI + np.arange(2, M_1 + 1)/(YY[i]*WN)
        OI      = DI[1:M_1]*MRR + DR[1:M_1] *MRI
        OR1     = OR*PSI[1:M_1] - PSI[:M_1 - 1]
        OR2     = OR*PSI[1:M_1] - OI*HI[1:M_1] - PSI[:M_1 - 1]
        OI1     = OI*PSI[1:M_1]
        OI2     = OR*HI[1:M_1] + OI*PSI[1:M_1] - HI[:M_1 - 1]
        BR[1:M_1]   = (OR1*OR2 + OI1*OI2)/(OR2**2 + OI2**2)
        BI[1:M_1]   = (OR2*OI1 - OR1*OI2)/(OR2**2 + OI2**2)                                                           
#       YAR     = AR[J]**2 + AI[J]**2 + BR[J]**2 + BI[J]**2
        del OR, OI, OR1, OR2, OI1, OI2
## END OF COMPUTING THE MIE COEFFICIENTS
        
        # 20210428 debug
        CE = np.sum(coeff[1, :M_1]*(AR[:M_1] + BR[:M_1]))
        CS = np.sum(coeff[1, :M_1]*(AR[:M_1]**2 + AI[:M_1]**2 + \
                                    BR[:M_1]**2 + BI[:M_1]**2))
        AR[:M_1] = coeff[2, :M_1]*(AR[:M_1] + BR[:M_1])
        AI[:M_1] = coeff[2, :M_1]*(AI[:M_1] + BI[:M_1])
        BR[:M_1] = AR[:M_1] - 2*coeff[2, :M_1]*BR[:M_1]
        BI[:M_1] = AI[:M_1] - 2*coeff[2, :M_1]*BI[:M_1]
        # CE = 0.0
        # CS = 0.0
        # for J in range(1, M_1+1):
        #     CJ = coeff[1, J-1]
        #     ARJ = AR[J-1]
        #     AIJ = AI[J-1]
        #     BRJ = BR[J-1]
        #     BIJ = BI[J-1]
        #     CDA = ARJ**2 + AIJ**2
        #     CDB = BRJ**2 + BIJ**2
        #     CE += CJ*(ARJ + BRJ)
        #     CS += CJ*(CDA + CDB)
        #     CJ = coeff[2, J-1]
        #     AR[J-1] = CJ*(ARJ + BRJ)
        #     AI[J-1] = CJ*(AIJ + BIJ)
        #     BR[J-1] = CJ*(ARJ - BRJ)
        #     BI[J-1] = CJ*(AIJ - BIJ)
        
        # 20210428 debuging
        C_ext   += WY[i]*CE
        C_sca   += WY[i]*CS
        PTaunArr = np.array(list(map(Angl, [M_1]*NG, X, np.tile(coeff[0, :], (NG, 1)))))
        SPR     = np.sum(AR[:M_1]*(PTaunArr[:,0] + PTaunArr[:,1]), axis = 1)
        SPI     = np.sum(AI[:M_1]*(PTaunArr[:,0] + PTaunArr[:,1]), axis = 1)
        SMR     = np.sum(BR[:M_1]*(PTaunArr[:,1] - PTaunArr[:,0]), axis = 1)
        SMI     = np.sum(BI[:M_1]*(PTaunArr[:,1] - PTaunArr[:,0]), axis = 1)
        D1      = (SPR**2 + SPI**2)*WY[i]
        D2      = (SMR**2 + SMI**2)*WY[i]
        F_11    += D1 + D2
        F_33    += D1 - D2
        F_12    += (SPR*SMR + SPI*SMI)*WY[i]*2.0
        F_34    += (SPR*SMI - SPI*SMR)*WY[i]*2.0
        del SPR, SPI, SMR, SMI, D1, D2
        
        time.sleep(0.0005)
        pass
    
## END OF AVERAGING OVER SIZES
    
## TODO: ELEMENTS OF THE SCATTERING MATRIX
    # 20210420 debug
    F_11 *= 2.0/C_sca
    F_12 *= 2.0/C_sca
    F_33 *= 2.0/C_sca
    F_34 *= 2.0/C_sca
        
## TODO: CROSS SECTIONS AND SINGLE SCATTERING ALBEDO
    # 20210420 debug
    C_ext  *= 2.0*np.pi/WN**2
    C_sca  *= 2.0*np.pi/WN**2
    Alb    = C_sca/C_ext
    
## TODO: CALCULATION OF THE EXPANSION COEFFICIENTS
    # 20210421 debug
    # coef = np.zeros([NPL, 8])
    arr1 = np.arange(2, L1_max)
    coef = np.zeros([L1_max-3+1, 8])
    coef[:, 0] = 1.0/(arr1 + 1.0)
    coef[:, 1] = 2.0*arr1 + 1.0
    coef[:, 2] = 1.0/np.sqrt((arr1 + 1.0)**2 - 4.0)
    coef[:, 3] = np.sqrt(arr1**2 - 4.0)
    coef[:, 4] = 1.0/(arr1*((arr1 + 1.0)**2 - 4.0))
    coef[:, 5] = (2.0*arr1 + 1.0)*(arr1*(arr1 + 1.0))
    coef[:, 6] = (2.0*arr1 + 1.0)*4.0
    coef[:, 7] = (arr1 + 1)*(arr1**2 - 4.0)
    
    # 20210421 debug
    alpha, beta = np.zeros([4, L1_max]), np.zeros([2, L1_max])
    L1_arr      = np.ones([NG, 1])*L1_max
    P           = np.array(list(map(Gener, X, L1_arr, np.tile(coef, [NG, 1, 1]))))
    ##F_11*W扩充为347*348
    alpha[0, :] = np.sum(F_11*W*P[:,:,0].T, axis = 1)
    alpha[1, :] = np.sum((F_11 + F_33)*W*P[:,:,1].T, axis = 1)
    alpha[2, :] = np.sum((F_11 - F_33)*W*P[:,:,2].T, axis = 1)
    alpha[3, :] = np.sum(F_33*W*P[:,:,0].T, axis = 1)
    beta[0, :]  = np.sum(F_12*W*P[:,:,3].T, axis = 1)
    beta[1, :]  = np.sum(F_34*W*P[:,:,3].T, axis = 1)
    del L1_arr, F_11, F_12, F_33, F_34
#    for i in range(NG):
#        P  = Gener(X[i], L1_max, coef)
#        alpha[0, :] += F_11[i]*W[i]*P[:, 0]
#        alpha[1, :] += (F_11[i] + F_33[i])*W[i]*P[:, 1]
#        alpha[2, :] += (F_11[i] - F_33[i])*W[i]*P[:, 2]
#        alpha[3, :] += F_33[i]*W[i]*P[:, 3]
#        beta[0, :] += F_12[i]*W[i]*P[:, 3]
#        beta[1, :] += F_34[i]*W[i]*P[:, 3]
    
    # 20210423 debug
    arr_tmp = np.abs(alpha[0, :]*(np.arange(L1_max) + 0.5))
    try:
        ind_tmp = np.argwhere(arr_tmp <= DDELT)[0]
        L1_max  = int(ind_tmp + 1)
    except:
        L1_max  = int(L1_max)
    CL_arr  = np.arange(L1_max) + 0.5
    alpha[0,:L1_max] *= CL_arr
    alpha[1,:L1_max] = 0.5*CL_arr*np.sum(alpha[1:3, :L1_max], axis = 0)
    alpha[2,:L1_max] = 0.5*CL_arr*np.diff(alpha[1:3, :L1_max], axis = 0)
    alpha[3,:L1_max] *= CL_arr
    beta[0,:L1_max]  *= CL_arr
    beta[1,:L1_max]  *= -CL_arr    
#    for i in range(L1_max):
#        CL  = i + 0.5
#        L = i + 1
#        alpha[0, i] *= CL
#        alpha[1, i] = 0.5*CL*np.sum(alpha[1:3, i])
#        alpha[2, i] = 0.5*CL*np.diff(alpha[1:3, i])
#        alpha[3, i] *= CL
#        beta[0, i]  *= CL
#        beta[1, i]  *= -CL
#        if np.abs(alpha[0, i]) <= DDELT:
#            break
#    L1_max = L
    
## TODO: Print out
    # 20210423 debug
    print('R_1 = ' + str(R_1) + ', R_2 = ' + str(R_2))
    print('R_eff = ' + str(Reff) + ', V_eff = ' + str(Veff))
    print('LAM = ' + str(Lam) + ', MRR = ' + str(MRR) + ', MRI = ' + str(MRI))
    print('NK = ' + str(NK) + ', N = ' + str(N) + ', NP = ' + str(NP))
    Q1 = alpha[0, 1]/3.0 # the ensemble-averaged asymmetry parameter
    print('<COS> = ' + str(Q1))
    print('C_ext = ' + str(C_ext) + ', C_sca = ' + str(C_sca) + ', Albedo = ' + str(Alb))
    print('Maximal order of Mie coefficients = ' + str(M))
    print()
    print('*********   EXPANSION COEFFICIENTS   *********')
    print('   S     ALPHA 1    ALPHA 2    ALPHA 3    ALPHA 4     BETA 1     BETA 2')
    for i in range(L1_max):
        print(str(i) + ', ' + str(alpha[0, i]) + ', ' + str(alpha[1, i]) + ', ' + str(alpha[2, i]) + \
              ', ' + str(alpha[3, i]) + ', ' + str(beta[0, i]) + ', ' + str(beta[1, i]))
    
    return C_ext, C_sca, area, Volume, Rvw, Rmean, alpha, beta, L1_max
    

def Mie(NDISTR = 2, AA = 0.10, BB = (np.log(1.05)**2), AA1 = None, AA2 = None, \
        BB1 = None, BB2 = None, GAM = 1.0, Lam = 0.865, MRR = 1.28, MRI = -0.03, \
        NK = 100, N = 100, NP = 0, R_1 = 0.001, R_2 = 20.0, DDELT = 1e-6):

    C_ext, C_sca, area, Volume, Rvw, Rmean, alpha, beta, L1 = Spher(AA, BB, GAM, Lam, MRR, MRI, \
                                                R_1, R_2, N, NP, NDISTR, NK, AA1, BB1, \
                                                AA2, BB2, DDELT)
    '''
    Reff: the effective radius of the size distribution.
    Veff: the effective variance of the size distribution.
    Rmean: the average radius. (5.252)
    Rvw: the volume-weighted average radius. (5.253)
    '''
    Q_ext = C_ext/area                              # Optical efficiency factors Q_ext
    Q_sca = C_sca/area
    L_max = L1 - 1                                  # Number of coefficients minus 1
    mat_sca = CalScaMatrix(alpha, beta, L_max)      # The scattering matrix
    
    return Q_ext, Q_sca, mat_sca
    














    
    
    
    
    
    
    
    
    
    
    
    
    
    