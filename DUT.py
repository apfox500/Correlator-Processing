import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from Constants import *
from utils import process_noise_data, linear


def dut_main(date:str, **kwargs) ->list[str]:
    filepaths = []
    # -------------------- Initialization --------------------
    num_samples_test = kwargs.get("num_samples_test", NUM_SAMPLES_TEST)
    data_file_base = kwargs.get("data_file_base", DUT_FILENAME).format(date=date)
    fft_freq = np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9))

    # -------------------- Process DUT Files and Calculate Average PSD --------------------
    ch1_avg_psd, ch2_avg_psd, csd_avg = process_noise_data(
        data_file_base=data_file_base,
        num_samples_test=num_samples_test,
        fft_freq=fft_freq
    ) # psd is returned in v^2/Hz

    # -------------------- Load Gain Data --------------------
    # Load gain data from CSV file

    gain_file = kwargs.get("gain_file", GAIN_FILE)
    if not os.path.exists(gain_file):
        raise FileNotFoundError(f"Gain file '{gain_file}' does not exist.")
    
    headers = kwargs.get("headers", GAIN_HEADERS)
    
    gain_df = pd.read_csv(gain_file) # headers should be something like: ['Freq','CW_Ch1','CW_Ch2','CW_Phase_Diff']

    # resample gain_df to match the frequency bins of the PSD data
    fft_freq = fft_freq[(fft_freq >= 1e9) & (fft_freq <= 2e9)]  # Trim to 1-2 GHz

    gain_freq = np.linspace(1, 2, 201)  # Assuming gain_df is in GHz
    gain_ch1 = np.interp(fft_freq / 1e9, gain_freq, gain_df[headers[1]])
    gain_ch2 = np.interp(fft_freq / 1e9, gain_freq, gain_df[headers[2]])
    gain_phase_diff = np.interp(fft_freq / 1e9, gain_freq, gain_df[headers[3]])

    # convert gain from dB to linear scale
    gain_ch1_linear = linear(gain_ch1)
    gain_ch2_linear = linear(gain_ch2)

    # -------------------- NT Calculation --------------------
    # convert to w/Hz
    ch1_avg_psd_w = ch1_avg_psd / IMPEDANCE  # Convert from V^2/Hz to W/Hz
    ch2_avg_psd_w = ch2_avg_psd / IMPEDANCE

    # Calculate NT for ch1 and ch2
    ch1_nt = ch1_avg_psd_w / (gain_ch1_linear * BOLTZ)
    ch2_nt = ch2_avg_psd_w / (gain_ch2_linear * BOLTZ)

    # -------------------- NF Calculation --------------------
    # Calculate NF for ch1 and ch2
    ch1_nf = 10 * np.log10(ch1_nt / T_AMB)
    ch2_nf = 10 * np.log10(ch2_nt / T_AMB)

    # -------------------- Angle Difference Calculation --------------------
    # Calculate the phase difference in radians
    dut_phase_diff = np.angle(csd_avg)

    # subtract the phase difference from the gain phase difference
    phase_diff = dut_phase_diff - gain_phase_diff
    
    # -------------------- Save + Plot Results --------------------
    #save to csv
    results_df = pd.DataFrame({
        'Frequency (GHz)': fft_freq / 1e9,
        'Ch1 NF (dB)': ch1_nf,
        'Ch2 NF (dB)': ch2_nf,
        'Phase Difference (rad)': phase_diff,
    })
    filepaths.append(os.path.join(OUT_DIR, f"DUT_NF_Calculation_{date}.csv"))
    results_df.to_csv(filepaths[-1], index=False)

    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot Noise Figure
    ax1.plot(fft_freq / 1e9, ch1_nf, label='Ch1 NF (dB)', color=COLORS[0])
    ax1.plot(fft_freq / 1e9, ch2_nf, label='Ch2 NF (dB)', color=COLORS[1])
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Noise Figure (dB)")
    ax1.legend(loc="center left")

    # Plot Phase Difference (degrees) on right axis
    phase_diff_deg = np.degrees(np.unwrap(phase_diff))
    ax2.plot(fft_freq / 1e9, phase_diff_deg, label='Phase Difference', color=COLORS[2])
    ax2.set_ylabel("Phase Difference (degrees)")
    ax2.legend(loc="center right")

    plt.title("DUT Noise Figure and Phase Difference vs Frequency")
    plt.tight_layout()
    filepaths.append(os.path.join(OUT_DIR, f"DUT_NF_{date}.png"))
    plt.savefig(filepaths[-1], bbox_inches='tight')
    plt.show()

    return filepaths
