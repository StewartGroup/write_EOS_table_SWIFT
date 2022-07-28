# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:19:43 2022

@author: adriana

CURRENTLY SET UP FOR PYROLITE
"""

#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import unyt
from numba import njit
#import h5py

import os, sys

import swiftsimio as sw
import woma
from woma.misc import io
from woma.eos import tillotson, sesame, idg, hm80
from woma.eos.T_rho import T_rho
from woma.misc import glob_vars as gv
from woma.misc import utils as ut


this_dir, this_file = os.path.split(__file__)
path = os.path.join(this_dir, "../aneos-pyrolite-2022/")
sys.path.append(path)
import eostable as eost

# ====>>>>>> YOU NEED TO MAKE SURE THESE VALUES MATCH ANEOS.INPUT  <<<<=====
MATERIALNAME = 'ANEOS pyrolite'
MODELNAME = 'Pyrolite_ANEOS_SLVTv0.2'
# Header information must all be compatible with float format
MATID   = 1.0      # MATID number
DATE    = 20210627 # Date as a single 8-digit number YYMMDD
VERSION = 0.1      # ANEOS Parameters Version number
FMN     = 153      # Formula weight in atomic numbers for NaCa2Fe4Mg30Al3Si24O89 left out Na and added 1 Si
FMW     = 3234.61  # Formula molecular weight (g/mol) for NaCa2Fe4Mg30Al3Si24O89 left out Na and added 1 Si
# The following define the default initial state for material in the 201 table
R0REF   = 3.35     # g/cm3 *** R0REF is inserted into the density array
K0REF   = 0.95E12  # dynes/cm2
T0REF   = 298.     # K -- *** T0REF is inserted into the temperature array
P0REF   = 1.E6     # dynes/cm2 -- this defines the principal Hugoniot calculated below
#

# MATERIALNAME = 'ANEOS forsterite'
# MODELNAME = 'Forsterite-ANEOS-SLVTv1.0G1'
# # Header information must all be compatible with float format
# MATID = 1.0        # MATID number
# DATE = 20220714    # Date as a single 8-digit number YYMMDD
# VERSION = 0.1      # ANEOS Parameters Version number
# FMN = 70.          # Formula weight in atomic numbers for Mg2SiO4
# FMW = 140.691      # Formula molecular weight (g/cm3) for Mg2SiO4
# # The following define the default initial state for material in the 201 table
# R0REF   = 3.22     # g/cm3 *** R0REF is inserted into the density array
# K0REF   = 1.1E12   # dynes/cm2
# T0REF   = 298.     # K -- *** T0REF is inserted into the temperature array
# P0REF   = 1.E6     # dynes/cm2 -- this defines the principal Hugoniot calculated below

## ==============================================================##

def write_table_SWIFT(Fp_table, name, version_date, A1_rho, A1_T, A2_u, A2_P, A2_c, A2_s, new=1):
    """Write the data to a file, in a SESAME-like format plus header info, etc.
    Adapted from sesame.write_table_SESAME
    
    File contents
    -------------
    
    # NEW header - AFTER 7/15/2022 (12 lines):
    version_date                                                (YYYYMMDD)
    num_rho  num_T
    rho[0]   rho[1]  ...  rho[num_rho]                          (kg/m^3)
    T[0]     T[1]    ...  T[num_T]                              (K)
    u[0, 0]                 P[0, 0]     c[0, 0]     s[0, 0]     (J/kg, Pa, m/s, J/K/kg)
    u[1, 0]                 ...         ...         ...
    ...                     ...         ...         ...
    u[num_rho-1, 0]         ...         ...         ...
    u[0, 1]                 ...         ...         ...
    ...                     ...         ...         ...
    u[num_rho-1, num_T-1]   ...         ...         s[num_rho-1, num_T-1]
    -------------
    
    # OLD header - BEFORE 7/15/2022 (6 lines):
     file version date
     ndensities  ntemperatures
     density array/(kg/m^3)
     Temperature array/K
     u/(J/kg)  p/Pa  cs/(km/s)  s/(J/kg/K) - inner loop over density
    -------------
    
    Parameters
    ----------
    Fp_table : str
        The table file path.
    name : str
        The material name.
    version_date : int
        The file version date (YYYYMMDD).
        Must match sesame.h SESAME_params objects in SWIFT?????????
    A1_rho, A1_T : [float]
        Density (kg m^-3) and temperature (K) arrays.
    A2_u, A2_P, A2_c, A2_s : [[float]]
        Table arrays of sp. int. energy (J kg^-1), pressure (Pa), sound speed
        (m s^-1), and sp. entropy (J K^-1 kg^-1).
    new : int (1)
        whether or not to use the new header and spacing (new=1) or old header and spaccing (otherwise), default is new
    """
    Fp_table = ut.check_end(Fp_table, ".txt")
    num_rho = len(A1_rho)
    num_T = len(A1_T)
    
    with open(Fp_table, "w") as f:
        # Header
        if new==1: 
            f.write("# Material %s\n" % name)
            f.write(
                "# version_date                                                (YYYYMMDD)\n"
                "# num_rho  num_T\n"
                "# rho[0]   rho[1]  ...  rho[num_rho-1]                        (kg/m^3)\n"
                "# T[0]     T[1]    ...  T[num_T-1]                            (K)\n"
                "# u[0, 0]                 P[0, 0]     c[0, 0]     s[0, 0]     (J/kg, Pa, m/s, J/K/kg)\n"
                "# u[1, 0]                 ...         ...         ...\n"
                "# ...                     ...         ...         ...\n"
                "# u[num_rho-1, 0]         ...         ...         ...\n"
                "# u[0, 1]                 ...         ...         ...\n"
                "# ...                     ...         ...         ...\n"
                "# u[num_rho-1, num_T-1]   ...         ...         s[num_rho-1, num_T-1]\n"
            )
        else:
            f.write(" # Material %s\n" % name)
            f.write(
                " # file version date\n"
                " # ndensities  ntemperatures\n"
                " # density array/(kg/m^3)\n"
                " # Temperature array/K\n"
                " # u/(J/kg)  p/Pa  cs/(km/s)  s/(J/kg/K) - inner loop over density\n"
            )
        
        # Metadata
        f.write("%d \n" % version_date)
        if new==1:
            f.write("%d %d \n" % (num_rho, num_T))
        else:
            f.write(" %d %d\n" % (num_rho, num_T))
        
        # Density and temperature arrays
        for i_rho in range(num_rho):
            if new==1:
                f.write("%.8e " % A1_rho[i_rho])
            else:
                f.write(" %.8e" % A1_rho[i_rho])
        f.write("\n")
        for i_T in range(num_T):
            if new==1:
                f.write("%.8e " % A1_T[i_T])
            else:
                f.write(" %.8e" % A1_T[i_T])
        f.write("\n")
        
        # Table arrays
        # old swift tables use weird formatting - i'm not going to try to replicate exactly
        if new==1:    
            for i_T in range(num_T):
                for i_rho in range(num_rho):
                    f.write(
                        "%.8e %.8e %.8e %.8e \n"
                        % (
                            A2_u[i_rho, i_T],
                            A2_P[i_rho, i_T],
                            A2_c[i_rho, i_T],
                            A2_s[i_rho, i_T],
                        )
                    )
                    
        else:    
            for i_T in range(num_T):
                for i_rho in range(num_rho):
                    f.write(
                        " %.8e %.8e %.8e %.8e\n"
                        % (
                            A2_u[i_rho, i_T],
                            A2_P[i_rho, i_T],
                            A2_c[i_rho, i_T],
                            A2_s[i_rho, i_T],
                        )
                    )
                
        print('Wrote file to ',Fp_table)
    
def convert_values(A1_rho, A2_u, A2_P, A2_c, A2_s, new=1):
    """
    Converts between ANEOS SESAME-style table output units and SWIFT input units
    Notes: prior to 7/15/2022 A2_c was in km/s, was changed to m/s in newer builds/branches

    Parameters
    ----------
    A1_rho : float
        Density (g/cm^3), from either -STD-NOTENSION or -EXT table
    A2_u : float
        Sp. internal energy (MJ/kg), from -STD-NOTENSION table
    A2_P : float
        Pressure (GPa), from -STD-NOTENSION table
    A2_c : float
        Sound speed (cm/s), from -EXT table
    A2_s : float
        Sp. entropy (MJ/kg/K), from -EXT table
    new : int, optional
        Whether to use new sound speed units. The default is 1, using new m/s

    Returns
    -------
    A1_rho_new : float
        Density (in kg/m^3)
    A2_u_new : float
        Sp. internal energy (in J/kg)
    A2_P_new : float
        Pressure (in Pa)
    A2_c_new : float
        Sound speed (in m/s (new) or km/s (old))
    A2_s_new : float
        Sp. entropy (in J/kg/K)

    """
    A1_rho_new = A1_rho*(1.E3)
    A2_u_new = A2_u*(1.E6)
    A2_P_new = A2_P*(1.E9)
    A2_s_new = A2_s*(1.E6)
    if new==1:
        A2_c_new = A2_c*(1.E-2)
    else:
        A2_c_new = A2_c*(1.E-5)
        
    return A1_rho_new, A2_u_new, A2_P_new, A2_c_new, A2_s_new

## ============================================================== ##

## MAKE SURE TO SET CORRECT TABLE PROPERTIES UP TOP ##

NewEOS = eost.extEOStable()  # FIRST make new empty EOS object
NewEOS.loadextsesame(
        path + "NEW-SESAME-EXT.TXT"
    )  # LOAD THE EXTENDED 301 SESAME FILE GENERATED BY STSM VERSION OF ANEOS
NewEOS.loadstdsesame(
        path + "NEW-SESAME-STD-NOTENSION.TXT"
    )  # LOAD THE STANDARD 301 SESAME FILE GENERATED BY STSM VERSION OF ANEOS
NewEOS.MODELNAME = MODELNAME  # string set above in user input
NewEOS.MDQ = np.zeros((NewEOS.NT, NewEOS.ND))  # makes the empty MDQ array
    # Units: g/cm3, K, GPa, MJ/kg, MJ/kg, MJ/K/kg, cm/s, MJ/K/kg, KPA flag. 2D arrays are (NT,ND).

    # Add the header info to the table. This could be done during the loading.
    # if made from this notebook, these values are set in the user-input above.
    # ** MAKE SURE THEY MATCH ANEOS.INPUT **
NewEOS.MATID = MATID
NewEOS.DATE = DATE
NewEOS.VERSION = VERSION
NewEOS.FMN = FMN
NewEOS.FMW = FMW
NewEOS.R0REF = R0REF
NewEOS.K0REF = K0REF
NewEOS.T0REF = T0REF
NewEOS.P0REF = P0REF

## Prepping data for conversion and dimension checks
n_rho = NewEOS.ND
n_T = NewEOS.NT
A1_rho = NewEOS.rho
A1_T = NewEOS.T
A2_u = np.transpose(NewEOS.U)
A2_P = np.transpose(NewEOS.P)
A2_c = np.transpose(NewEOS.cs)
A2_s = np.transpose(NewEOS.S)

new = 0
A1_rho, A2_u, A2_P, A2_c, A2_s = convert_values(A1_rho, A2_u, A2_P, A2_c, A2_s, new=0)

# validate shapes of arrays and whatnot
sesame.prepare_table_SESAME(A1_rho, A1_T, A2_P, A2_u, A2_s, verbosity=1)

filename = 'ANEOS_pyrolite_S22_old'  # write file function will append filetype specifier
filepath = './'+filename
write_table_SWIFT(filepath, MATERIALNAME, DATE, A1_rho, A1_T, A2_u, A2_P, A2_c, A2_s, new=0)