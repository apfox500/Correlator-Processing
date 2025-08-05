import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

from numpy.typing import NDArray

from Constants import *
from utils import process_noise_data, parse_complex

def XParameters(dut_s11:NDArray, dut_s11_conj:NDArray, dut_s21:NDArray, dut_s21_conj:NDArray, dut_s22:NDArray, dut_s22_conj:NDArray, # DUT S-parameters
                s11:NDArray, s66:NDArray, s66_conj:NDArray, # S-parameters of Correlator
                s31:NDArray, s46:NDArray, s46_conj:NDArray, # Gain of Correlator
                dut_b3:NDArray, dut_b4:NDArray, dut_b3_b4_conj:NDArray, # DUT PSDs in W/Hz
                load_b3:NDArray, load_b4:NDArray, # Load Cal PSDs in W/Hz
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Calculate X-parameters for the DUT based on the provided S-parameters and PSDs.

    Parameters
    ----------
        dut_s11 : NDArray
            S11 parameter of the DUT.
        dut_s11_conj : NDArray
            Conjugate of S11 parameter of the DUT.
        dut_s21 : NDArray
            S21 parameter of the DUT.
        dut_s21_conj : NDArray
            Conjugate of S21 parameter of the DUT.
        dut_s22 : NDArray
            S22 parameter of the DUT.
        dut_s22_conj : NDArray
            Conjugate of S22 parameter of the DUT.
        s11 : NDArray
            S11 parameter of the correlator (reflectance Ch1).
        s66 : NDArray
            S66 parameter of the correlator(reflectance Ch2).
        s66_conj : NDArray
            Conjugate of S66 parameter of the correlator.
        s31 : NDArray
            Channel 1 gain.
        s46 : NDArray
            Gchannel 2 gain.
        s46_conj : NDArray
            Conjugate Channel 2 gain.
        dut_b3 : NDArray
            PSD from channel 1 in W/Hz.
        dut_b4 : NDArray
            PSD from channel 2 in W/Hz.
        dut_b3_b4_conj : NDArray
            Cross PSD between channels 1 and 2 in W/Hz.
        load_b3 : NDArray
            50 Ohm Load calibration PSD for channel 1 in W/Hz.
        load_b4 : NDArray
            50 Ohm Load calibration PSD for channel 2 in W/Hz.

    Returns
    -------
        xn1_sq : NDArray
            <|xn1|^2> - Noise power in channel 1
        xn2_sq : NDArray
            <|xn2|^2> - Noise power in channel 2
        xn1_xn2_conj : NDArray
            <xn1*xn2_conj> - Cross noise term
    """

    # calculate <|xn1|^2>
    term1_num = np.abs(1 - dut_s11 * s11)**2
    term1_den = np.abs(s31)**2
    b3_diff = dut_b3 - load_b3
    
    xn1_sq = (term1_num / term1_den) * b3_diff + (term1_num - np.abs(dut_s11)**2) * BOLTZ * T_AMB

    # calculate <|xn2|^2>
    term2_num = np.abs(1 - dut_s22 * s66)**2
    term2_den = np.abs(s46)**2
    b4_diff = dut_b4 - load_b4
    s21_term = np.abs(dut_s21)**2 / np.abs(1 - dut_s11 * s11)**2

    xn2_sq = (term2_num / term2_den) * b4_diff + (term2_num - np.abs(dut_s22)**2 - s21_term) * BOLTZ * T_AMB

    # calculate <xn1*xn2_conj>
    term3_num = (1 - dut_s11 * s11) * (1 - dut_s22_conj * s66_conj)
    term3_den = s31 * s46_conj
    
    term4_num = dut_s11 * dut_s21_conj
    term4_den = 1 - dut_s11_conj * s11 

    xn1_xn2_conj = (term3_num / term3_den) * dut_b3_b4_conj - (term4_num / term4_den) * BOLTZ * T_AMB

    return xn1_sq, xn2_sq, xn1_xn2_conj


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

    fft_freq = fft_freq[(fft_freq >= 1e9) & (fft_freq <= 2e9)]

    dut_b3 = ch1_avg_psd / IMPEDANCE
    dut_b4 = ch2_avg_psd / IMPEDANCE
    dut_b3_b4_conj = csd_avg / IMPEDANCE


    # -------------------- Load Gain Data --------------------
    # Load gain data from CSV file

    gain_file = kwargs.get("gain_file", GAIN_FILE)
    if not os.path.exists(gain_file):
        raise FileNotFoundError(f"Gain file '{gain_file}' does not exist.")
    
    gain_headers = kwargs.get("gain_headers", GAIN_HEADERS) # should be like [Freq, S31, S46]

    gain_df = pd.read_csv(gain_file, converters={gain_headers[1]: parse_complex, gain_headers[2]: parse_complex})

    # Interpolate the complex gain values to match the PSD frequency bins
    s31 = np.interp(fft_freq, gain_df['Frequency'] * 1e9, gain_df[gain_headers[1]])
    s46 = np.interp(fft_freq, gain_df['Frequency'] * 1e9, gain_df[gain_headers[2]])
    s46_conj = np.conj(s46)

    # -------------------- Load Load Cal Data --------------------
    load_file = kwargs.get("load_file", LOAD_FILE)
    load_headers = kwargs.get("load_headers", LOAD_HEADERS)
    if not os.path.exists(load_file):
        raise FileNotFoundError(f"Load file '{load_file}' does not exist.")

    load_df = pd.read_csv(load_file, converters={load_headers[1]: parse_complex, load_headers[2]: parse_complex}) # headers should be something like ['Freq', 'Ch1_b', 'Ch2_b', 'Phase']

    # resample load_df to match the frequency bins of the PSD data
    load_freq = np.linspace(1, 2, 3001)
    load_b3 = np.interp(fft_freq / 1e9, load_freq, load_df[load_headers[1]]) / IMPEDANCE # ch1 load PSD in v^2/Hz
    load_b4 = np.interp(fft_freq / 1e9, load_freq, load_df[load_headers[2]]) / IMPEDANCE # ch2 load PSD

    # -------------------- Load relevant s-parameters --------------------
    # ---- load DUT s-params ----
    dut_s_file = kwargs.get("dut_s_file", DUT_S_FILE)
    if not os.path.exists(dut_s_file):
        raise FileNotFoundError(f"DUT s-parameter file '{dut_s_file}' does not exist.")

    dut_ntwk = rf.Network(dut_s_file)
    # Resample DUT network to match PSD frequency bins
    interp_freqs = np.round(fft_freq, 6)
    dut_ntwk = dut_ntwk.interpolate(interp_freqs)  # skrf expects GHz

    # Extract S-parameters
    dut_s11 = dut_ntwk.s[:, 0, 0]
    dut_s21 = dut_ntwk.s[:, 1, 0]
    dut_s22 = dut_ntwk.s[:, 1, 1]

    dut_s11_conj = np.conj(dut_s11)
    dut_s21_conj = np.conj(dut_s21)
    dut_s22_conj = np.conj(dut_s22)

    # ---- load s11 and 2 66 ----
    s11_file = kwargs.get("s11_file", S11_FILE)
    s66_file = kwargs.get("s66_file", S66_FILE)

    if not os.path.exists(s11_file):
        raise FileNotFoundError(f"S11 file '{s11_file}' does not exist.")

    if not os.path.exists(s66_file):
        raise FileNotFoundError(f"S66 file '{s66_file}' does not exist.")
    
    s11_ntwk = rf.Network(s11_file)
    s66_ntwk = rf.Network(s66_file)

    # Interpolate to match PSD frequency bins (in GHz)
    s11_ntwk = s11_ntwk.interpolate(interp_freqs)
    s66_ntwk = s66_ntwk.interpolate(interp_freqs)

    # Extract s11 and s66 from their respective s1p files
    s11 = s11_ntwk.s[:, 0, 0]
    s66 = s66_ntwk.s[:, 0, 0]

    s11_conj = np.conj(s11)
    s66_conj = np.conj(s66)


    # -------------------- Calculate X-parameters --------------------

    xn1_sq, xn2_sq, xn1_xn2_conj = XParameters(
        dut_s11, dut_s11_conj, dut_s21, dut_s21_conj, dut_s22, dut_s22_conj,
        s11, s66, s66_conj,
        s31, s46, s46_conj,
        dut_b3, dut_b4, dut_b3_b4_conj,
        load_b3, load_b4
    )
    
    
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot |xn1|^2 and |xn2|^2 (magnitude, left axis)
    ax1.plot(fft_freq / 1e9, np.abs(xn1_sq), label='|xn1|^2', color=COLORS[0])
    ax1.plot(fft_freq / 1e9, np.abs(xn2_sq), label='|xn2|^2', color=COLORS[1])
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (V$^2$/Hz)")
    ax1.legend(loc="upper left")

    # Plot angle of xn1_xn2_conj (degrees, right axis)
    angle_xn1_xn2 = np.degrees(np.unwrap(np.angle(xn1_xn2_conj)))
    ax2.plot(fft_freq / 1e9, angle_xn1_xn2, label='∠(xn1·xn2*)', color=COLORS[2])
    ax2.set_ylabel("Angle (degrees)")
    ax2.legend(loc="upper right")

    plt.title("X-parameters vs Frequency")
    plt.tight_layout()
    plt.show()

    # Plot |xn1|^2 and |xn2|^2 (magnitude, left axis)

    # -------------------- NT Calculation --------------------
    
    
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
