
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- User Configurable Section ---
CSV_PATH = "correlator_testing_output/Processed_Pwr_vs_f_PSplitterNominalCWs-30dBm_Att20dB_25-07-16-16-22-50.csv"
COLORS = ["#223A64", "#fa753c", "#ffda49", "#a9d574", "#3ca590", "#626c9d"]
TITLE = "Average Gain"
XLABEL = "Frequency (GHz)"
YLABEL_GAIN = "Gain (dBm)"
YLABEL_PHASE = "Phase Difference (degrees)"
SAVEFIG = False
OUT_PATH = "correlator_testing_output/.png"

# --- Load Data ---
df = pd.read_csv(CSV_PATH)

# --- Prepare Data ---
freq = df["ch1_real_freq"] / 1e9  # GHz
ch1_gain = df["ch1_gain"]
ch2_gain = df["ch2_gain"]
phase_diff = np.degrees(np.unwrap(df["phase_diff"]))

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

# Plot gains
ax1.plot(freq, ch1_gain, '-', color=COLORS[4], label='Ch1 Gain')
ax1.plot(freq, ch2_gain, '-', color=COLORS[1], label='Ch2 Gain')
ax1.set_xlabel(XLABEL)
ax1.set_ylabel(YLABEL_GAIN)
ax1.legend(loc="center left")

# Plot phase difference
ax2.plot(freq, phase_diff, '-', color=COLORS[3], label='Phase Difference')
ax2.set_ylabel(YLABEL_PHASE)
ax2.legend(loc="center right")

plt.title(TITLE)
plt.tight_layout()
if SAVEFIG:
    plt.savefig(OUT_PATH, bbox_inches='tight')
plt.show()
