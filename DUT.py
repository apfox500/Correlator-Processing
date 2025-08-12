import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

from numpy.typing import NDArray
from typing import Tuple


from Constants import *
from utils import *

def load_param_df(param_df, fft_freq, **kwargs):
    # -------------------- Load Gain Data --------------------
    # Load gain data from CSV file

    gain_file = kwargs.get("gain_file", GAIN_FILE)
    if not os.path.exists(gain_file):
        raise FileNotFoundError(f"Gain file '{gain_file}' does not exist.")
    
    gain_headers = kwargs.get("gain_headers", GAIN_HEADERS) # should be like [Freq, S31, S46]

    gain_df = pd.read_csv(gain_file, converters={gain_headers[1]: parse_complex, gain_headers[2]: parse_complex})

    s46 = gain_df[gain_headers[2]]
    s31 = gain_df[gain_headers[1]]

    # Add gain data to param_df
    param_df['s31'] = interp_complex(fft_freq, gain_df['Frequency'] * 1e9, s31)
    param_df['s46'] = interp_complex(fft_freq, gain_df['Frequency'] * 1e9, s46)
    param_df['s46_conj'] = np.conj(param_df['s46'])

    # -------------------- Load Load Cal Data --------------------
    load_file = kwargs.get("load_file", LOAD_FILE)
    load_headers = kwargs.get("load_headers", LOAD_HEADERS)
    if not os.path.exists(load_file):
        raise FileNotFoundError(f"Load file '{load_file}' does not exist.")

    load_df = pd.read_csv(load_file, converters={load_headers[1]: parse_complex, load_headers[2]: parse_complex}) # headers should be something like ['Freq', 'Ch1_b', 'Ch2_b', 'Phase']

    # resample load_df to match the frequency bins of the PSD data
    load_freq = np.linspace(1, 2, len(load_df)) 
    param_df['load_b3'] = np.interp(fft_freq, load_freq * 1e9, load_df[load_headers[1]]) / R_0 # ch1 load PSD in W/Hz
    param_df['load_b4'] = np.interp(fft_freq, load_freq * 1e9, load_df[load_headers[2]]) / R_0 # ch2 load PSD


    # -------------------- Load relevant s-parameters --------------------
    # ---- load DUT s-params ----
    dut_s_file = kwargs.get("dut_s_file", DUT_S_FILE)
    if not os.path.exists(dut_s_file):
        raise FileNotFoundError(f"DUT s-parameter file '{dut_s_file}' does not exist.")

    dut_ntwk = rf.Network(dut_s_file)
    # Resample DUT network to match PSD frequency bins
    dut_s11 = dut_ntwk.s[:, 0, 0]
    dut_s21 = dut_ntwk.s[:, 1, 0]
    dut_s22 = dut_ntwk.s[:, 1, 1]


    # Extract S-parameters
    param_df['dut_s11'] = interp_complex(fft_freq, dut_ntwk.f, dut_s11)
    param_df['dut_s21'] = interp_complex(fft_freq, dut_ntwk.f, dut_s21)
    param_df['dut_s22'] = interp_complex(fft_freq, dut_ntwk.f, dut_s22)

    param_df['dut_s11_conj'] = np.conj(param_df['dut_s11'])
    param_df['dut_s21_conj'] = np.conj(param_df['dut_s21'])
    param_df['dut_s22_conj'] = np.conj(param_df['dut_s22'])

    # ---- load s11 and s66 ----
    s11_file = kwargs.get("s11_file", S11_FILE)
    s66_file = kwargs.get("s66_file", S66_FILE)

    if not os.path.exists(s11_file):
        raise FileNotFoundError(f"S11 file '{s11_file}' does not exist.")

    if not os.path.exists(s66_file):
        raise FileNotFoundError(f"S66 file '{s66_file}' does not exist.")
    
    s11_ntwk = rf.Network(s11_file)
    s66_ntwk = rf.Network(s66_file)


    # Extract s11 and s66 from their respective s1p files
    s11_s = s11_ntwk.s[:, 0, 0]
    s66_s = s66_ntwk.s[:, 0, 0]


    # Store the interpolated values in the DataFrame
    param_df['s11'] = interp_complex(fft_freq, s11_ntwk.f, s11_s)
    param_df['s66'] = interp_complex(fft_freq, s66_ntwk.f, s66_s)

    param_df['s11_conj'] = np.conj(param_df['s11'])
    param_df['s66_conj'] = np.conj(param_df['s66'])

    return param_df

def XParameters(param_df: pd.DataFrame) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Calculate X-parameters for the DUT based on the provided S-parameters and PSDs.

    Parameters
    ----------
    param_df : pd.DataFrame
        DataFrame containing the following columns:
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
    term1_num = np.abs(1 - param_df['dut_s11'] * param_df['s11'])**2
    term1_den = np.abs(param_df['s31'])**2
    b3_diff = np.abs(param_df['dut_b3']) - np.abs(param_df['load_b3'])
    
    xn1_sq = (term1_num / term1_den) * b3_diff + (term1_num - np.abs(param_df['dut_s11'])**2) * KB * T_AMB

    # calculate <|xn2|^2>
    term2_num = np.abs(1 - param_df['dut_s22'] * param_df['s66'])**2
    term2_den = np.abs(param_df['s46'])**2
    b4_diff = np.abs(param_df['dut_b4']) - np.abs(param_df['load_b4'])
    s21_term = np.abs(param_df['dut_s21'])**2 / np.abs(1 - param_df['dut_s11'] * param_df['s11'])**2

    xn2_sq = (term2_num / term2_den) * b4_diff + (term2_num - np.abs(param_df['dut_s22'])**2 - s21_term) * KB * T_AMB

    # calculate <xn1*xn2_conj>
    term3_num = (1 - param_df['dut_s11'] * param_df['s11']) * (1 - param_df['dut_s22_conj'] * param_df['s66_conj'])
    term3_den = param_df['s31'] * param_df['s46_conj']

    term4_num = param_df['dut_s11'] * param_df['dut_s21_conj']
    term4_den = 1 - param_df['dut_s11_conj'] * param_df['s11_conj']

    # Calculate term3_num / term3_den and term4_num / term4_den
    term3 = term3_num / term3_den
    term4 = term4_num / term4_den

    xn1_xn2_conj = term3 * param_df['dut_b3_b4_conj'] - term4 * KB * T_AMB

    return xn1_sq, xn2_sq, xn1_xn2_conj

def NoiseParameters(x1:NDArray[np.float64], x2:NDArray[np.float64], x12:NDArray[np.complex128], s11, gamma_G = 0) -> pd.DataFrame:
    # Solve for t (noise temp)
    t = x1 + (np.abs(1 + s11)**2) * x2 - 2 * np.real(np.conj(1+s11) * x12)

    # solve for eta
    eta_num = x2*(1+np.abs(s11)**2)  + x1 - 2 * np.real(np.conj(s11) * x12)
    eta_den = x2*s11 - x12
    eta = eta_num / eta_den
    
    # solve for gamma opt (optimal reflection coefficient)
    radical = 1 - 4 / np.abs(eta)**2
    gamma_opt = eta / 2 * (1-np.sqrt(radical))

    # solve for T min (minimum noise temperature)
    t_min_num = x2 - np.abs(gamma_opt)**2 * (x1 + np.abs(s11)**2*x2 - 2 * np.real(np.conj(s11) * x12))
    t_min_den = 1 + np.abs(gamma_opt)**2
    t_min = t_min_num / t_min_den

    # Solve for Te (effective noise temp)
    Te_num = np.abs(gamma_opt - gamma_G)**2
    Te_den = np.abs(1+gamma_opt)**2 * (1 - np.abs(gamma_G)**2)
    Te = t_min + t *Te_num / Te_den

    return pd.DataFrame({
        'T_min': t_min,
        'gamma_opt': gamma_opt,
        'eta': eta,
        't': t,
        'gamma_G': gamma_G,
        'Te': Te,
    })

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

    param_df = pd.DataFrame({
        'Freq (GHz)': fft_freq / 1e9,
        'dut_b3': ch1_avg_psd / R_0,  # Convert to W/Hz
        'dut_b4': ch2_avg_psd / R_0,  # Convert to W/Hz
        'dut_b3_b4_conj': csd_avg / R_0  # Convert to W/Hz
    })    

    # -------------------- Load Gain, Load Cal, and S parameter Data --------------------

    param_df = load_param_df(param_df, fft_freq, **kwargs)

    # -------------------- Calculate (and plot) X-parameters --------------------

    xn1_sq, xn2_sq, xn1_xn2_conj = XParameters(param_df)
    
    # Plot |xn1|^2 and |xn2|^2 (in dB/Hz, left axis)
    if(kwargs.get("graph_x", True)):
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(fft_freq / 1e9, dB(np.abs(xn1_sq)), label=r'$|\langle x_{n1}^2\rangle |$', color=COLORS[0])
        ax1.plot(fft_freq / 1e9, dB(np.abs(xn2_sq)), label=r'$|\langle x_{n2}^2\rangle |$', color=COLORS[1])
        ax1.plot(fft_freq / 1e9, dB(np.abs(xn1_xn2_conj)), label=r'$|\langle x_{n1}\cdot x_{n2}^{\bf{*}}\rangle |$', color=COLORS[3])
        ax1.set_xlabel("Frequency (GHz)")
        ax1.set_ylabel("Magnitude (dB/Hz)")
        ax2.plot(fft_freq / 1e9, np.unwrap(np.degrees(np.angle(xn1_xn2_conj))), label=r'$\angle \langle x_{n1}\cdot x_{n2}^{\bf{*}} \rangle $', color=COLORS[4], linestyle=':')
        ax2.set_ylabel("Phase (degrees)")
        
        # Combine legends from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right")
        plt.title("X-parameters Magnitude and Phase vs Frequency")
        plt.tight_layout()
        filepaths.append(os.path.join(OUT_DIR, f"XParameters_{date}.png"))
        plt.savefig(filepaths[-1])
        plt.show()

    # -------------------- NT Calculation --------------------
    # convert to Kelvin
    x1 = xn1_sq / KB
    x2 = xn2_sq / np.abs(param_df['dut_s21'])**2 / KB
    x12 = xn1_xn2_conj / np.conj(param_df['dut_s21'])  / KB

    noise_params= NoiseParameters(x1, x2, x12, param_df['dut_s11'], gamma_G=kwargs.get("gamma_G", 0))
    
    if kwargs.get("graph_eta", True):
        plt.figure(figsize=(12, 6))
        plt.plot(fft_freq / 1e9, np.abs(noise_params['eta']), color=COLORS[0], label=r'$|\eta|$')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel(r"$|\eta|$")
        plt.title(r"$|\eta|$ vs Frequency")
        plt.legend()
        plt.tight_layout()
        filepaths.append(os.path.join(OUT_DIR, f"Eta_vs_Freq_{date}.png"))
        plt.savefig(filepaths[-1])
        plt.show()
    
    # calculate Rn (real impedance)
    Rn = noise_params['t']/(4*296.15)*R_0 # real impedance in Ohms
    
    if kwargs.get("graph_rn", True):
        plt.figure(figsize=(12, 6))
        plt.plot(fft_freq / 1e9, Rn, color=COLORS[0], label=r'$R_n$')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel(r"$R_n$ (Ohms)")
        plt.title(r"$R_n$ vs Frequency")
        plt.legend()
        plt.tight_layout()
        filepaths.append(os.path.join(OUT_DIR, f"Rn_vs_Freq_{date}.png"))
        plt.savefig(filepaths[-1])
        plt.show()

    if kwargs.get("graph_gamma_opt", True):
        # graph phase and magnitude of gamma_opt
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(fft_freq / 1e9, dB(np.abs(noise_params['gamma_opt'])), color=COLORS[0], label=r'$|\Gamma_{opt}|$')
        ax2.plot(fft_freq / 1e9, np.unwrap(np.degrees(np.angle(noise_params['gamma_opt']))), color=COLORS[1], label=r'$\angle \Gamma_{opt}$')
        ax1.set_xlabel("Frequency (GHz)")
        ax1.set_ylabel(r"$|\Gamma_{opt} (dB ?)|$")
        ax2.set_ylabel(r"$\angle \Gamma_{opt}$ (degrees)")
        plt.title(r"$\Gamma_{opt}$ vs Frequency")
        # Combine legends from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right")
        plt.tight_layout()
        filepaths.append(os.path.join(OUT_DIR, f"GammaOpt_vs_Freq_{date}.png"))
        plt.savefig(filepaths[-1])
        plt.show()
    
    if kwargs.get("graph_t_min", True):
        plt.figure(figsize=(12, 6))
        plt.plot(fft_freq / 1e9, noise_params['T_min'], color=COLORS[0], label=r'$T_{min}$')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel(r"$T_{min}$ (K)")
        plt.title(r"$T_{min}$ vs Frequency")
        plt.legend()
        plt.tight_layout()
        filepaths.append(os.path.join(OUT_DIR, f"Tmin_vs_Freq_{date}.png"))
        plt.savefig(filepaths[-1])
        plt.show()

    if kwargs.get("graph_te", True):
        plt.figure(figsize=(12, 6))
        plt.plot(fft_freq / 1e9, noise_params['Te'], color=COLORS[0], label=r'$T_e$')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel(r"$T_e$ (K)")
        plt.title(r"$T_e$ vs Frequency")
        plt.legend()
        plt.tight_layout()
        filepaths.append(os.path.join(OUT_DIR, f"Te_vs_Freq_{date}.png"))
        plt.savefig(filepaths[-1])
        plt.show()
    
    # -------------------- Calculate and plot Noise Figure --------------------
    # Calculate noise figure

    nf = 10 * np.log10((T_AMB + noise_params['Te']) / T_AMB)

    if kwargs.get("graph_nf", True):
        #plot noise figure
        plt.figure(figsize=(12, 6))
        plt.plot(fft_freq / 1e9, nf, color=COLORS[0], label=r'$NF$')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel(r"$NF$ (dB)")
        plt.title(r"$NF$ vs Frequency")
        plt.legend()
        plt.tight_layout()
        filepaths.append(os.path.join(OUT_DIR, f"NF_vs_Freq_{date}.png"))
        plt.savefig(filepaths[-1])
        plt.show()
    # -------------------- Save results --------------------
    xparams_df = pd.DataFrame({
        'Frequency (GHz)': fft_freq / 1e9,
        'xn1_sq': xn1_sq,
        'xn2_sq': xn2_sq,
        'xn1_xn2_conj': xn1_xn2_conj,
        'Te': noise_params['Te'],
        'T_min': noise_params['T_min'],
        'gamma_opt': noise_params['gamma_opt'],
        'gamma_G': noise_params['gamma_G'],
        't': noise_params['t'],
        'Rn': Rn,
        'NF': nf,

    })
    filepaths.append(f"Processed_DUT_{date}.csv")
    xparams_df.to_csv(filepaths[-1], index=False)

    return filepaths