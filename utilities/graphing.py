import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

COLORS = ["#223A64","#fa753c", "#ffda49", "#a9d574", "#3ca590", "#626c9d"]



freq = np.linspace(1, 2, 3001)  # Frequency range from 1 GHz to 2 GHz

def angle(x):
    return np.unwrap(np.degrees(np.unwrap(np.angle(x))))

def dB(value) :
    return 10 * np.log10(value) if np.all(value > 0) else np.where(value > 0, 10 * np.log10(value), -np.inf)

def parse_complex(s):
    if pd.isna(s) or s == '':
        return np.nan
    return complex(s.strip('()'))

# read in csv with specified headers and convert to complex dtype
headers = [
    "one_minus_dut_s11_s11",
    "one_minus_dut_s22conj_s66conj",
    "term3_num",
    "term3_den",
    "term4_num",
    "term4_den",
    "term3",
    "term4",
    "xn1_xn2_conj"
]
df = pd.read_csv(
    "correlator_testing_output/XParameters_debug.csv",
    names=headers,
    header=0,
    converters={col: parse_complex for col in headers}
)


def plot(x, x_name):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.plot(freq, dB(np.abs(x)), color=COLORS[0], label='Magnitude')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Phase (degrees)')
    ax2.plot(freq, angle(x), color=COLORS[0], label='Phase', linestyle='--')
    ax2.tick_params(axis='y')
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f"{x_name} Response")
    plt.show()


def plot2(x, y, x_name, y_name):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.plot(freq, dB(np.abs(x)), color=COLORS[0], label=f"{x_name} Magnitude")
    ax1.plot(freq, dB(np.abs(y)), color=COLORS[1], label=f"{y_name} Magnitude")
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Phase (degrees)')
    ax2.plot(freq, angle(x), color=COLORS[0], label=f"{x_name} Phase", linestyle='--')
    ax2.plot(freq, angle(y), color=COLORS[1], label=f"{y_name} Phase", linestyle='--')
    ax2.tick_params(axis='y')
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f"{x_name} and {y_name} Comparison")
    plt.show()


def plots(names):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Phase (degrees)')

    for i, name in enumerate(names):
        color = COLORS[i % len(COLORS)]
        ax1.plot(freq, dB(np.abs(df[name])), color=color, label=f"{name} Magnitude")
        ax2.plot(freq, angle(df[name]), color=color, linestyle='--', label=f"{name} Phase")

    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title("Comparison of Selected Responses")
    fig.tight_layout()
    plt.show()

plots(["term3", "term3_num", "xn1_xn2_conj", "term3_den"])