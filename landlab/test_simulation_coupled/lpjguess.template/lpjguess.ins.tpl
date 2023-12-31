!
! Instruction file with global PFTs to be used with the
! demonstration I/O module.
!
! See the file data/env/readme_env.txt for information about the
! data files.
!
! Make sure to start LPJ-GUESS with -input demo when using this
! instruction file.
!

! FOCUSSITES
!site	name	lat	lon	elevation
!1	Nahuelbuta	-37.80904	-73.01382	1230
!2	LaCampana	-32.92905	-71.08846	375
!3	SantaGracia	-29.75214	-71.1575	579
!4	PandeAzucar	-26.11008	-70.55039	328

! model run definitions
import "earthshape_lpjsetup.ins"

! pft definitions
import "earthshape_pfts.ins"

! outputdirectory is mandatory. Should be "./" for parallel runs.
outputdirectory "./output/"


!///////////////////////////////////////////////////////////////////////////////////////
! Forcing Data & gridlists
!
! Atmospheric CO2 content (ppmv)
! (used only by guessio.cpp)
param "file_co2"      (str "input/climdata/co2_TraCE_egu2018_35ka_const180ppm.txt")

! N deposition (blank string to use constant pre-industrial level of 2 kgN/ha/year)
param "file_temp"     (str "input/climdata/$CLIMTEMP")
param "variable_temp" (str "temp")

param "file_prec"     (str "input/climdata/$CLIMPREC")
param "variable_prec" (str "prec")

param "file_wetdays"  (str "input/climdata/$CLIMWET") 
param "variable_wetdays" (str "wet")

param "file_insol"      (str "input/climdata/$CLIMRAD") 
param "variable_insol"  (str "rad")

param "file_min_temp"      (str "")
param "variable_min_temp"  (str "min_temp")

param "file_max_temp"      (str "")
param "variable_max_temp"  (str "max_temp")

param "file_ndep"     (str "../forcings/climdata/ndepo_lamarque_cl_hist_landid.nc")
param "variable_nhxdry" (str "NHxDry")
param "variable_nhxwet" (str "NHxWet")
param "variable_noydry" (str "NOyDry")
param "variable_noywet" (str "NOyWet")

! landform data
param "file_landform" (str "input/lfdata/$LFDATA")
param "variable_fraction" (str "fraction")
param "variable_elevation" (str "elevation")    ! also use for site data file
param "variable_slope" (str "slope")
param "variable_asp_slope" (str "asp_slope")
param "variable_aspect" (str "aspect")

! THIS SHOULD BE OPTIONAL SO WE CAN STILL RUN DEFAULT LPJ INVENTORY RUNS !!!
param "variable_soildepth" (str "soildepth")


! site data
param "file_site" (str "input/lfdata/$SITEDATA")
param "variable_soc" (str "soc")
param "variable_clay" (str "clay")
param "variable_silt" (str "silt")
param "variable_sand" (str "sand")
! global soil depth/ constant for now
!param "variable_depth" (str "depth")

param "file_gridlist_cf" (str "$GRIDLIST")


!///////////////////////////////////////////////////////////////////////////////
! SERIALIZATION SETTINGS
!///////////////////////////////////////////////////////////////////////////////
nyear_spinup $NYEARSPINUP ! number of years to spin up the simulation for
!state_year 600			! year to save/start state file (no setting = after spinup)
                        ! state_year for first segment, otherwise 100 (?)
restart $RESTART				! whether to start from a state file

! save_state in run_landform always 0 as we auto-save into landform_state_path
save_state 0			! whether to save a state file
state_path "loaddir/"	! directory to read state files from
 
landform_state_path "dumpdir_eor/"	! directory to put state files in 

