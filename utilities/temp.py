import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = ["#223A64","#fa753c", "#ffda49", "#a9d574", "#3ca590", "#626c9d"]

def angle(x):
    return np.unwrap(np.degrees(np.unwrap(np.angle(x))))

def dB(value):
    return 10 * np.log10(value) if np.all(value > 0) else np.where(value > 0, 10 * np.log10(value), -np.inf)

def parse_complex(s):
    if pd.isna(s) or s == '':
        return np.nan
    return complex(s.strip('()'))

# Read the CSV data
df = pd.read_csv(
    "./correlator_testing_output/CW_Complex_Gain_25-07-16-16-22-50.csv",
    converters={'S31': parse_complex, 'S46': parse_complex}
)

# Extract frequency and complex values
freq = df['Frequency'].values
S31 = df['S31'].values
S46 = df['S46'].values

# Calculate S31 * conj(S46)
S31_S46_conj = S31 * np.conj(S46)

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel('Magnitude (dB)')
ax1.plot(freq, dB(np.abs(S31_S46_conj)), color=COLORS[0], label='Magnitude')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel('Phase (degrees)')
ax2.plot(freq, angle(S31_S46_conj), color=COLORS[0], label='Phase', linestyle='--')
ax2.tick_params(axis='y')
fig.tight_layout()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("S31 * conj(S46) Response")
plt.show()