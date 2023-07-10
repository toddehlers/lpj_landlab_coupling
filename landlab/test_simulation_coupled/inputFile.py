"""
This file contains the input-parameters for the landlab driver. 

This does NOT use the landlab inherent load_params method but declares
variables directly in python and is loaded in the driver file with

	from inputFile.py import *

This was done because we do not use the standard-inputs for the
fluvial and hillslope routines but do the processing for
vegetation influence directly in the driver.

Usage:

	-Parameternames must be equal to the ones declared in the 
	driver file (better not change at all.)
	-Comments can be written in a standart python way
	-Any declaration of the same variable in the driver-file
	will overwrite the imported values so code-with-caution.


Created by: Manuel Schmid, May 28th, 2018
"""

#Model Grid Parameters
ncols = 21	#number of columns
nrows = 21 #number of rows
dx    = 100 #spacing between nodes

#Model Runtime Parameters
totalT = 1e6 #total model runtime
ssT    = 1e6  #spin-up time before sin-modulation, set to same value as totalT for steady-state-simulations
sfT    = 1e6  #spin-up time before step-change-modulation, set to same value as totalT for steady-state-simulations
spin_up = 0.9e6 
dt = 100

#Uplift
upliftRate = 1.e-5 #m/yr, Topographic uplift rate

#Surface Processes
#Linear Diffusion:
linDiffBase = 1e-2 #m2/yr, base linear diffusivity for bare-bedrock
alphaDiff   = 0.3  #Scaling factor for vegetation-influence (see Instabulluoglu and Bras 2005)

#Fluvial Erosion:
critArea    = 1e6 #L^2, Minimum Area which the steepness-calculator assumes for channel formation.
aqDens      = 1000 #Kg/m^3, density of water
grav        = 9.81 #m/s^2, acceleration of gravity
nSoil       = 0.01 #Mannings number for bare soil
nVRef       = 0.6  #Mannings number for reference vegetation
vRef        = 1    #1 = 100%, reference vegetation-cover for fully vegetated conditions
w           = 1    #Scaling factor for vegetation-influence (see Istanbulluoglu and Bras 2005)

#Fluvial Erosion/SPACE:
k_sediment = 6e-9 
k_bedrock  = 6e-10 
Ff         = 0 
phi        = 0.1
Hstar      = 10.
vs         = 5 
m          = 0.6
n          = 0.7
#sp_crit_sedi = 5.e-4
sp_crit_sedi = 5.e-4
#sp_crit_bedrock = 6.e-4
sp_crit_bedrock = 7.e-4
solver = 'adaptive'

#Lithology
initialSoilDepth = 1 #m
soilProductionRate = 0.0000006 #m/dt

#Climate Parameters
baseRainfall = float(3) #m/dt, base steady-state rainfall-mean over the dt-timespan
rfA          = 0 #m, rainfall-step-change if used

#Vegetation Cover
vp = .1 #initial vegetation cover, 1 = 100%
sinAmp = 0.1 #vegetation cover amplitude for oscillation
sinPeriod = 1e5 #yrs, period of sin-modification

#LPJ_coupling_parameters:
latitude   = -26.25 #center-coordinate of grid cell for model area
longitude  = -70.75 #center-coordinate of grid cell for model area
lpj_output = '../input/sp_lai.out'
LPJGUESS_INPUT_PATH = './temp_lpj'
LPJGUESS_TEMPLATE_PATH = './lpjguess.template'
LPJGUESS_FORCINGS_PATH = './forcings'
LPJGUESS_INS_FILE_TPL = 'lpjguess.ins.tpl'
LPJGUESS_BIN = '/esd/esd01/data/mschmid/coupling/build/guess'
LPJGUESS_CO2FILE = 'co2_TraCE_egu2018_35ka_const180ppm.txt'

#landform classifier input:
classificationType = 'SIMPLE'
elevationStepBin   = 200

#output
outInt = 100 #yrs, model-time-interval in which output is created
