import os
import math

# ---------------------------------------------------------------------------
# This Section should be changed every run, choosing the data to be processed
# ---------------------------------------------------------------------------

# Directories
DATA_DIR = "/Volumes/noise/CHIPS/Data/Data_AnalogDevice/July_2025"
OUT_DIR = "./correlator_testing_output"
CW_DIR = os.path.join(OUT_DIR, "CW")
ND_DIR = os.path.join(OUT_DIR, "NoiseDiode")
LOAD_DIR = os.path.join(OUT_DIR, "LoadCal")
DUT_DIR = os.path.join(OUT_DIR, "DUT")

# Dates + file formats
CW_DATE = "25-07-16-16-22-50"
CW_FILENAME = "f_CW{freq}GHz_PwrSensorOnCouplerZUDC20-5R23-S+PortB_Split02P1P2OnCorrelatorCh1Ch2_NSamp131072_PWR-30dBm_Att20dB_Ch115dBCh225dB__Fs{FS}GHz_{date}_0.npy"

ND_DATE = "25-07-16-09-12-45"
ND_FILENAME = "NoiseDiode136_AttCh115dBCh225dB_Splitter02_Port0_NSamp2097152_Att0dB__Fs10.0GHz_{date}"

LOAD_DATE = "25-07-16-09-25-10"
LOAD_FILENAME = "LoadCal_AttCh1_15dBCh2_25dB_NSamp2097152__Fs10.0GHz_{date}"

DUT_DATE = "25-07-22-12-17-17"
DUT_FILENAME = "DUT_ZKL2Plus_AttCh1_15dBCh2_25dB_NSamp2097152__Fs10.0GHz_{date}"

# CW
FILE1 = "alphabeta/1_Coupler01_1-2GHz_P_A and P_1_Adp_0 -ActuallyThisOneDoesNotHaveAdp.s2p"
FILE2 = "alphabeta/2_S11_S21_S22_PA_PB_By XifengLu.txt"
FILE3 = "alphabeta/3_Gamma_PowerMeterSenor_By XifengLu.txt"
FILE4 = "alphabeta/4_S11_CH1_ATT14dB.s1p"
FILE5 = "alphabeta/5_Coupler01_1-2GHz_P_A and P_6_0_DeEmbedAdpLu.s2p"
FILE6 = "alphabeta/6_S11_CH2_ATT26dB.s1p"

# Noise Diode
CAL_FILE = "alphabeta/Diode136_X1_cal_data.txt"
CH1_FILE = "alphabeta/ZPAD-23-S+_CommonP1_20241024 1.s2p"
CH2_FILE = "alphabeta/ZPAD-23-S+_CommonP2_20241024 1.s2p"

# Load and DUT
CW_GAIN_FILE = f"correlator_testing_output/CW/Processed_CW_{CW_DATE}.csv"
CW_GAIN_HEADERS = ['Frequency', 'S31', 'S46']

ND_GAIN_FILE = f"./correlator_testing_output/NoiseDiode/Processed_Noise_{ND_DATE}.csv"
ND_GAIN_HEADERS = ['Freq (GHz)', 'Ch1 Gain', 'Ch2 Gain', 'Phase Difference (rad)']

LOAD_FILE = f"./correlator_testing_output/LoadCal/Processed_Load_{LOAD_DATE}.csv"
LOAD_HEADERS = ['Freq', 'Ch1 PSD', 'Ch2 PSD']

# DUT only
DUT_S_FILE = "alphabeta/DUT1.s2p"
S11_FILE = "alphabeta/S11_CH1_ATT14dB.s1p"
S66_FILE = "alphabeta/S66_CH2_ATT26dB - Copy.s1p"


# ---------------------------------------------------------------------------
# This Section contains constants that shoudn't need to be changed every time
# ---------------------------------------------------------------------------


# ADC Settings
ADC_RESOLUTION = 12
VOLTAGE_RANGE = 5.0
VOLT_PER_TICK = VOLTAGE_RANGE / (2**ADC_RESOLUTION - 1)
FS = 10.0 # Sampling frequency in GHz

# Constants
SAMPLE_CUTOFF = 30000
MINIMUM_FREQUENCY = 1e9  # 1 GHz
MAXIMUM_FREQUENCY = 2e9
DATASET_LENGTH = math.floor(MAXIMUM_FREQUENCY * SAMPLE_CUTOFF / FS / 1e9) - math.ceil(MINIMUM_FREQUENCY * SAMPLE_CUTOFF / FS / 1e9) + 1
NUM_TRACES = 500 # max 500
COLORS = ["#223A64","#fa753c", "#ffda49", "#a9d574", "#3ca590", "#626c9d"]
KB = 1.380649e-23  # Boltzmann constant in J/K
T_AMB = 290 # Ambient temperature in K
R_0 = 50  # Ohms