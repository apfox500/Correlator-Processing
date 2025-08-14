import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf

from numpy.typing import NDArray
from typing import Tuple

from Constants import *
from utils import dB, interp_complex, parse_complex, process_noise_data

def load_param_df(param_df: pd.DataFrame, fft_freq: np.ndarray, gain_type: str = "CW", **kwargs) -> pd.DataFrame:
    """
    Load parameter DataFrame with gain data based on specified gain type.
    
    This function loads and interpolates gain data (CW or ND), load calibration data,
    and S-parameter data, then adds all parameters to the input DataFrame.
    
    Parameters:
        param_df (pd.DataFrame): Base DataFrame containing frequency and PSD data.
        fft_freq (np.ndarray): Frequency array in Hz.
        gain_type (str): Type of gain to load ("CW" or "ND"). Defaults to "CW".
        **kwargs: Additional keyword arguments.
            cw_gain_file (str): Path to CW gain file (default: CW_GAIN_FILE).
            cw_gain_headers (list): CW gain column headers (default: CW_GAIN_HEADERS).
            nd_gain_file (str): Path to ND gain file (default: ND_GAIN_FILE).
            nd_gain_headers (list): ND gain column headers (default: ND_GAIN_HEADERS).
            load_file (str): Path to load calibration file (default: LOAD_FILE).
            load_headers (list): Load calibration column headers (default: LOAD_HEADERS).
            dut_s_file (str): Path to DUT S-parameter file (default: DUT_S_FILE).
            s11_file (str): Path to S11 parameter file (default: S11_FILE).
            s66_file (str): Path to S66 parameter file (default: S66_FILE).
        
    Returns:
        pd.DataFrame: Parameter DataFrame with gain, load, and S-parameter data loaded.
        
    Raises:
        ValueError: If gain_type is not "CW" or "ND".
        FileNotFoundError: If required parameter files do not exist.
    """
    
    if gain_type.upper() == "CW":
        # load CW gain data
        cw_gain_file = kwargs.get("cw_gain_file", CW_GAIN_FILE)
        if not os.path.exists(cw_gain_file):
            raise FileNotFoundError(f"CW gain file '{cw_gain_file}' does not exist.")
        
        cw_gain_headers = kwargs.get("cw_gain_headers", CW_GAIN_HEADERS)  # [freq, S31, S46]

        cw_gain_df = pd.read_csv(cw_gain_file, converters={cw_gain_headers[1]: parse_complex, cw_gain_headers[2]: parse_complex})

        cw_s46 = cw_gain_df[cw_gain_headers[2]]
        cw_s31 = cw_gain_df[cw_gain_headers[1]]

        # add CW gain data to param_df with standard names
        param_df['s31'] = interp_complex(fft_freq, cw_gain_df[cw_gain_headers[0]] * 1e9, cw_s31)
        param_df['s46'] = interp_complex(fft_freq, cw_gain_df[cw_gain_headers[0]] * 1e9, cw_s46)
        param_df['s46_conj'] = np.conj(param_df['s46'])
        
    elif gain_type.upper() == "ND":
        # load ND gain data
        nd_gain_file = kwargs.get("nd_gain_file", ND_GAIN_FILE)
        if not os.path.exists(nd_gain_file):
            raise FileNotFoundError(f"ND gain file '{nd_gain_file}' does not exist.")
        
        nd_gain_headers = kwargs.get("nd_gain_headers", ND_GAIN_HEADERS)  # [freq, ch1_gain, ch2_gain, phase_diff]

        nd_gain_df = pd.read_csv(nd_gain_file, converters={nd_gain_headers[1]: parse_complex, nd_gain_headers[2]: parse_complex})

        # interpolate complex gains directly (no need to convert from dB and phase)
        nd_s31 = interp_complex(fft_freq, nd_gain_df[nd_gain_headers[0]] * 1e9, nd_gain_df[nd_gain_headers[1]])
        nd_s46 = interp_complex(fft_freq, nd_gain_df[nd_gain_headers[0]] * 1e9, nd_gain_df[nd_gain_headers[2]])

        # add ND gain data to param_df with standard names
        param_df['s31'] = nd_s31
        param_df['s46'] = nd_s46
        param_df['s46_conj'] = np.conj(param_df['s46'])
        
    else:
        raise ValueError(f"Invalid gain_type '{gain_type}'. Must be 'CW' or 'ND'.")

    # load load calibration data
    load_file = kwargs.get("load_file", LOAD_FILE)
    load_headers = kwargs.get("load_headers", LOAD_HEADERS)
    if not os.path.exists(load_file):
        raise FileNotFoundError(f"Load file '{load_file}' does not exist.")

    load_df = pd.read_csv(load_file, converters={load_headers[1]: parse_complex, load_headers[2]: parse_complex})

    # resample load_df to match the frequency bins of the PSD data
    load_freq = load_df[load_headers[0]]  # frequency column in GHz

    param_df['load_b3'] = np.interp(fft_freq, load_freq * 1e9, load_df[load_headers[1]]) / R_0  # ch1 load PSD in W/Hz
    param_df['load_b4'] = np.interp(fft_freq, load_freq * 1e9, load_df[load_headers[2]]) / R_0  # ch2 load PSD in W/Hz


    # load S-parameters
    # load DUT S-parameters
    dut_s_file = kwargs.get("dut_s_file", DUT_S_FILE)
    if not os.path.exists(dut_s_file):
        raise FileNotFoundError(f"DUT s-parameter file '{dut_s_file}' does not exist.")

    dut_ntwk = rf.Network(dut_s_file)
    # resample DUT network to match PSD frequency bins
    dut_s11 = dut_ntwk.s[:, 0, 0]
    dut_s21 = dut_ntwk.s[:, 1, 0]
    dut_s22 = dut_ntwk.s[:, 1, 1]


    # extract S-parameters
    param_df['dut_s11'] = interp_complex(fft_freq, dut_ntwk.f, dut_s11)
    param_df['dut_s21'] = interp_complex(fft_freq, dut_ntwk.f, dut_s21)
    param_df['dut_s22'] = interp_complex(fft_freq, dut_ntwk.f, dut_s22)

    param_df['dut_s11_conj'] = np.conj(param_df['dut_s11'])
    param_df['dut_s21_conj'] = np.conj(param_df['dut_s21'])
    param_df['dut_s22_conj'] = np.conj(param_df['dut_s22'])

    # load S11 and S66 parameters
    s11_file = kwargs.get("s11_file", S11_FILE)
    s66_file = kwargs.get("s66_file", S66_FILE)

    if not os.path.exists(s11_file):
        raise FileNotFoundError(f"S11 file '{s11_file}' does not exist.")

    if not os.path.exists(s66_file):
        raise FileNotFoundError(f"S66 file '{s66_file}' does not exist.")
    
    s11_ntwk = rf.Network(s11_file)
    s66_ntwk = rf.Network(s66_file)

    # extract S11 and S66 from their respective S1P files
    s11_s = s11_ntwk.s[:, 0, 0]
    s66_s = s66_ntwk.s[:, 0, 0]

    # store the interpolated values in the DataFrame
    param_df['s11'] = interp_complex(fft_freq, s11_ntwk.f, s11_s)
    param_df['s66'] = interp_complex(fft_freq, s66_ntwk.f, s66_s)

    param_df['s11_conj'] = np.conj(param_df['s11'])
    param_df['s66_conj'] = np.conj(param_df['s66'])

    return param_df

def XParameters(param_df: pd.DataFrame) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Calculate X-parameters for the DUT based on S-parameters and PSDs.

    This function computes the noise X-parameters using the measured PSDs
    and known S-parameters of the DUT and correlator system.

    Parameters:
        param_df (pd.DataFrame): DataFrame containing the following columns:
            - dut_s11, dut_s11_conj: S11 parameter of the DUT and its conjugate
            - dut_s21, dut_s21_conj: S21 parameter of the DUT and its conjugate  
            - dut_s22, dut_s22_conj: S22 parameter of the DUT and its conjugate
            - s11: S11 parameter of the correlator (reflectance Ch1)
            - s66, s66_conj: S66 parameter of the correlator (reflectance Ch2) and conjugate
            - s31: Channel 1 gain
            - s46, s46_conj: Channel 2 gain and its conjugate
            - dut_b3: PSD from channel 1 in W/Hz
            - dut_b4: PSD from channel 2 in W/Hz  
            - dut_b3_b4_conj: Cross PSD between channels 1 and 2 in W/Hz
            - load_b3: 50 Ohm load calibration PSD for channel 1 in W/Hz
            - load_b4: 50 Ohm load calibration PSD for channel 2 in W/Hz

    Returns:
        tuple: A tuple containing:
            - xn1_sq (NDArray): <|xn1|^2> - Noise power in channel 1
            - xn2_sq (NDArray): <|xn2|^2> - Noise power in channel 2  
            - xn1_xn2_conj (NDArray): <xn1*xn2_conj> - Cross noise term
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

def NoiseParameters(x1: NDArray[np.float64], x2: NDArray[np.float64], x12: NDArray[np.complex128], s11, gamma_G=0) -> pd.DataFrame:
    """
    Calculate noise parameters from X-parameters.
    
    This function computes the four noise parameters (minimum noise temperature,
    optimum reflection coefficient, noise resistance, and effective noise temperature)
    from the measured X-parameters.
    
    Parameters:
        x1 (NDArray): Normalized noise power in channel 1 (in Kelvin).
        x2 (NDArray): Normalized noise power in channel 2 (in Kelvin).
        x12 (NDArray): Normalized cross noise term (in Kelvin).
        s11: S11 parameter of the DUT.
        gamma_G (float): Generator reflection coefficient. Defaults to 0 (matched load).
        
    Returns:
        pd.DataFrame: DataFrame containing noise parameters:
            - T_min: Minimum noise temperature in Kelvin
            - gamma_opt: Optimum reflection coefficient (complex)
            - eta: Noise correlation coefficient
            - t: Noise temperature in Kelvin
            - gamma_G: Generator reflection coefficient  
            - Te: Effective noise temperature in Kelvin
    """
    # solve for t (noise temperature)
    t = x1 + (np.abs(1 + s11)**2) * x2 - 2 * np.real(np.conj(1+s11) * x12)

    # solve for eta (noise correlation coefficient)
    eta_num = x2*(1+np.abs(s11)**2) + x1 - 2 * np.real(np.conj(s11) * x12)
    eta_den = x2*s11 - x12
    eta = eta_num / eta_den
    
    # solve for gamma_opt (optimal reflection coefficient)
    radical = 1 - 4 / np.abs(eta)**2
    gamma_opt = eta / 2 * (1-np.sqrt(radical))

    # solve for T_min (minimum noise temperature)
    t_min_num = x2 - np.abs(gamma_opt)**2 * (x1 + np.abs(s11)**2*x2 - 2 * np.real(np.conj(s11) * x12))
    t_min_den = 1 + np.abs(gamma_opt)**2
    t_min = t_min_num / t_min_den

    # solve for Te (effective noise temperature)
    Te_num = np.abs(gamma_opt - gamma_G)**2
    Te_den = np.abs(1+gamma_opt)**2 * (1 - np.abs(gamma_G)**2)
    Te = t_min + t * Te_num / Te_den

    return pd.DataFrame({
        'T_min': t_min,
        'gamma_opt': gamma_opt,
        'eta': eta,
        't': t,
        'gamma_G': gamma_G,
        'Te': Te,
    })

def plot_x_parameters(fft_freq, xn1_sq, xn2_sq, xn1_xn2_conj, gain_type, date, output_dir):
    """
    Plot X-parameters magnitude and phase vs frequency.
    
    Parameters:
        fft_freq (np.ndarray): Frequency array in Hz.
        xn1_sq (np.ndarray): X-parameter <|xn1|^2>.
        xn2_sq (np.ndarray): X-parameter <|xn2|^2>.
        xn1_xn2_conj (np.ndarray): X-parameter <xn1*xn2_conj>.
        gain_type (str): Gain type ("CW" or "ND") for labeling.
        date (str): Date string for filename.
        output_dir (str): Output directory path.
        
    Returns:
        str: Path to the saved plot file.
    """
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(fft_freq / 1e9, dB(np.abs(xn1_sq)), label=r'$|\langle x_{n1}^2\rangle |$', color=COLORS[0])
    ax1.plot(fft_freq / 1e9, dB(np.abs(xn2_sq)), label=r'$|\langle x_{n2}^2\rangle |$', color=COLORS[1])
    ax1.plot(fft_freq / 1e9, dB(np.abs(xn1_xn2_conj)), label=r'$|\langle x_{n1}\cdot x_{n2}^{\bf{*}}\rangle |$', color=COLORS[3])
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB/Hz)")
    ax2.plot(fft_freq / 1e9, np.degrees(np.unwrap(np.angle(xn1_xn2_conj))), label=r'$\angle \langle x_{n1}\cdot x_{n2}^{\bf{*}} \rangle $', color=COLORS[4], linestyle=':')
    ax2.set_ylabel("Phase (degrees)")
    
    # combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right")
    plt.title(f"X-parameters Magnitude and Phase vs Frequency ({gain_type} Gain)")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"DUT_XParameters_{gain_type}_{date}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_x_parameters_comparison(fft_freq, xn1_sq_cw, xn2_sq_cw, xn1_xn2_conj_cw,
                                xn1_sq_nd, xn2_sq_nd, xn1_xn2_conj_nd, date, output_dir):
    """
    Plot comparison of X-parameters between CW and ND gains.
    
    Parameters:
        fft_freq (np.ndarray): Frequency array in Hz.
        xn1_sq_cw, xn2_sq_cw, xn1_xn2_conj_cw: CW X-parameters.
        xn1_sq_nd, xn2_sq_nd, xn1_xn2_conj_nd: ND X-parameters.
        date (str): Date string for filename.
        output_dir (str): Output directory path.
        
    Returns:
        str: Path to the saved plot file.
    """
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(fft_freq / 1e9, dB(np.abs(xn1_sq_cw)), label=r'CW: $|\langle x_{n1}^2\rangle |$', color=COLORS[0], linestyle='-')
    ax1.plot(fft_freq / 1e9, dB(np.abs(xn1_sq_nd)), label=r'ND: $|\langle x_{n1}^2\rangle |$', color=COLORS[0], linestyle='--')
    ax1.plot(fft_freq / 1e9, dB(np.abs(xn2_sq_cw)), label=r'CW: $|\langle x_{n2}^2\rangle |$', color=COLORS[1], linestyle='-')
    ax1.plot(fft_freq / 1e9, dB(np.abs(xn2_sq_nd)), label=r'ND: $|\langle x_{n2}^2\rangle |$', color=COLORS[1], linestyle='--')
    ax1.plot(fft_freq / 1e9, dB(np.abs(xn1_xn2_conj_cw)), label=r'CW: $|\langle x_{n1}\cdot x_{n2}^{\bf{*}}\rangle |$', color=COLORS[3], linestyle='-')
    ax1.plot(fft_freq / 1e9, dB(np.abs(xn1_xn2_conj_nd)), label=r'ND: $|\langle x_{n1}\cdot x_{n2}^{\bf{*}}\rangle |$', color=COLORS[3], linestyle='--')
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB/Hz)")
    ax2.plot(fft_freq / 1e9, np.degrees(np.unwrap(np.angle(xn1_xn2_conj_cw))), label=r'CW: $\angle \langle x_{n1}\cdot x_{n2}^{\bf{*}} \rangle $', color=COLORS[4], linestyle='-')
    ax2.plot(fft_freq / 1e9, np.degrees(np.unwrap(np.angle(xn1_xn2_conj_nd))), label=r'ND: $\angle \langle x_{n1}\cdot x_{n2}^{\bf{*}} \rangle $', color=COLORS[4], linestyle='--')
    ax2.set_ylabel("Phase (degrees)")
    
    # combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right")
    plt.title("X-parameters Comparison: CW vs ND Gain")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"DUT_XParameters_Comparison_{date}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_single_parameter(fft_freq, parameter, param_name, unit, gain_type, date, output_dir):
    """
    Plot a single parameter vs frequency.
    
    Parameters:
        fft_freq (np.ndarray): Frequency array in Hz.
        parameter (np.ndarray): Parameter values to plot.
        param_name (str): Parameter name for title and filename.
        unit (str): Unit string for y-axis label.
        gain_type (str): Gain type ("CW" or "ND") for labeling.
        date (str): Date string for filename.
        output_dir (str): Output directory path.
        
    Returns:
        str: Path to the saved plot file.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(fft_freq / 1e9, parameter, color=COLORS[0], label=f'${param_name}$')
    plt.xlabel("Frequency (GHz)")
    plt.ylabel(f"${param_name}$ ({unit})")
    plt.title(f"${param_name}$ vs Frequency ({gain_type} Gain)")
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"DUT_{param_name}_{gain_type}_{date}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_complex_parameter(fft_freq, parameter, param_name, mag_unit, gain_type, date, output_dir):
    """
    Plot magnitude and phase of a complex parameter vs frequency.
    
    Parameters:
        fft_freq (np.ndarray): Frequency array in Hz.
        parameter (np.ndarray): Complex parameter values to plot.
        param_name (str): Parameter name for title and filename.
        mag_unit (str): Unit string for magnitude y-axis label.
        gain_type (str): Gain type ("CW" or "ND") for labeling.
        date (str): Date string for filename.
        output_dir (str): Output directory path.
        
    Returns:
        str: Path to the saved plot file.
    """
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(fft_freq / 1e9, dB(np.abs(parameter)), color=COLORS[0], label=f'$|{param_name}|$')
    ax2.plot(fft_freq / 1e9, np.unwrap(np.degrees(np.angle(parameter))), color=COLORS[1], label=f'$\\angle {param_name}$')
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel(f'$|{param_name}|$ ({mag_unit})')
    ax2.set_ylabel(f'$\\angle {param_name}$ (degrees)')
    plt.title(f'${param_name}$ vs Frequency ({gain_type} Gain)')
    # combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"DUT_{param_name}_{gain_type}_{date}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def plot_parameter_comparison(fft_freq, param_cw, param_nd, param_name, unit, date, output_dir):
    """
    Plot comparison of a parameter between CW and ND gains.
    
    Parameters:
        fft_freq (np.ndarray): Frequency array in Hz.
        param_cw (np.ndarray): CW parameter values.
        param_nd (np.ndarray): ND parameter values.
        param_name (str): Parameter name for title and filename.
        unit (str): Unit string for y-axis label.
        date (str): Date string for filename.
        output_dir (str): Output directory path.
        
    Returns:
        str: Path to the saved plot file.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(fft_freq / 1e9, param_cw, color=COLORS[0], linestyle='-', label=f'CW: ${param_name}$')
    plt.plot(fft_freq / 1e9, param_nd, color=COLORS[0], linestyle='--', label=f'ND: ${param_name}$')
    plt.xlabel("Frequency (GHz)")
    plt.ylabel(f'${param_name}$ ({unit})')
    plt.title(f'{param_name} Comparison: CW vs ND Gain')
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"DUT_{param_name}_Comparison_{date}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def dut_main(date: str = DUT_DATE, **kwargs) -> list[str]:
    """
    Main function for Device Under Test (DUT) noise parameter extraction and analysis.
    
    Processes correlator data to extract noise parameters including minimum noise temperature,
    optimal source reflection coefficient, noise figure, and noise resistance. Supports both
    continuous wave (CW) and noise diode (ND) gain calibration methods.
    
    Parameters:
        date (str, optional): Date string for file identification. Defaults to DUT_DATE.
        **kwargs: Additional configuration options:
            gamma_G (complex, optional): Generator reflection coefficient. Defaults to 0.
            graph_s31_s46_phase (bool, optional): Plot S31*S46 phase. Defaults to False.
            graph_x (bool, optional): Plot X-parameters. Defaults to True.
            graph_eta (bool, optional): Plot eta parameter. Defaults to True.
            graph_rn (bool, optional): Plot noise resistance. Defaults to True.
            graph_gamma_opt (bool, optional): Plot optimal reflection coefficient. Defaults to True.
            graph_t_min (bool, optional): Plot minimum noise temperature. Defaults to True.
            graph_te (bool, optional): Plot equivalent noise temperature. Defaults to True.
            graph_nf (bool, optional): Plot noise figure. Defaults to True.
            
    Returns:
        list[str]: List of file paths to generated plots and saved CSV data.
        
    Notes:
        - Loads correlator measurement data from specified date directory
        - Applies gain and S-parameter corrections for both CW and ND methods
        - Calculates X-parameters from corrected correlator data
        - Extracts noise parameters using standard noise theory
        - Generates comparative plots between CW and ND calibration methods
        - Saves processed data to CSV file for further analysis
    """
    # ensure output directory exists
    if not os.path.exists(DUT_DIR):
        os.makedirs(DUT_DIR)
    filepaths = []
    # -------------------- Initialization --------------------
    num_traces = kwargs.get("num_traces", NUM_TRACES)
    filename = kwargs.get("filename", DUT_FILENAME).format(date=date)
    fft_freq = np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9))

    # -------------------- Process DUT Files and Calculate Average PSD --------------------
    ch1_avg_psd, ch2_avg_psd, csd_avg = process_noise_data(
        data_file_base=filename,
        num_traces=num_traces,
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

    # Create parameter DataFrames for both CW and ND gain types
    param_df_cw = load_param_df(param_df.copy(), fft_freq, gain_type="CW", **kwargs)
    param_df_nd = load_param_df(param_df.copy(), fft_freq, gain_type="ND", **kwargs)

    if kwargs.get("graph_s31_s46_phase", False):
        plt.figure(figsize=(12, 6))
        plt.plot(
            fft_freq / 1e9,
            np.degrees(np.unwrap(np.angle(param_df_cw['s31'] * param_df_cw['s46_conj']))),
            color=COLORS[0],
            label="CW: phase(s31 * s46_conj)"
        )
        plt.plot(
            fft_freq / 1e9,
            np.degrees(np.unwrap(np.angle(param_df_nd['s31'] * param_df_nd['s46_conj']))),
            color=COLORS[1],
            label="ND: phase(s31 * s46_conj)"
        )
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Phase (degrees)")
        plt.title("Phase of s31 * s46_conj vs Frequency")
        plt.legend()
        plt.tight_layout()
        filepaths.append(os.path.join(DUT_DIR, f"DUT_S31_S46_Phase_{date}.png"))
        plt.savefig(filepaths[-1])
        plt.close()

    # -------------------- Calculate X-parameters for both gain types --------------------

    xn1_sq_cw, xn2_sq_cw, xn1_xn2_conj_cw = XParameters(param_df_cw)
    xn1_sq_nd, xn2_sq_nd, xn1_xn2_conj_nd = XParameters(param_df_nd)
    
    # Plot |xn1|^2 and |xn2|^2 (in dB/Hz, left axis) - CW version
    if kwargs.get("graph_x", True):
        filepaths.append(plot_x_parameters(fft_freq, xn1_sq_cw, xn2_sq_cw, xn1_xn2_conj_cw, "CW", date, DUT_DIR))
        filepaths.append(plot_x_parameters(fft_freq, xn1_sq_nd, xn2_sq_nd, xn1_xn2_conj_nd, "ND", date, DUT_DIR))
        filepaths.append(plot_x_parameters_comparison(fft_freq, xn1_sq_cw, xn2_sq_cw, xn1_xn2_conj_cw,
                                                     xn1_sq_nd, xn2_sq_nd, xn1_xn2_conj_nd, date, DUT_DIR))

    # -------------------- NT Calculation for both gain types --------------------
    # convert to Kelvin - CW
    x1_cw = xn1_sq_cw / KB
    x2_cw = xn2_sq_cw / np.abs(param_df_cw['dut_s21'])**2 / KB
    x12_cw = xn1_xn2_conj_cw / np.conj(param_df_cw['dut_s21'])  / KB

    noise_params_cw = NoiseParameters(x1_cw, x2_cw, x12_cw, param_df_cw['dut_s11'], gamma_G=kwargs.get("gamma_G", 0))
    
    # convert to Kelvin - ND
    x1_nd = xn1_sq_nd / KB
    x2_nd = xn2_sq_nd / np.abs(param_df_nd['dut_s21'])**2 / KB
    x12_nd = xn1_xn2_conj_nd / np.conj(param_df_nd['dut_s21'])  / KB

    noise_params_nd = NoiseParameters(x1_nd, x2_nd, x12_nd, param_df_nd['dut_s11'], gamma_G=kwargs.get("gamma_G", 0))
    
    if kwargs.get("graph_eta", True):
        filepaths.append(plot_single_parameter(fft_freq, np.abs(noise_params_cw['eta']), 'eta', '', "CW", date, DUT_DIR))
        filepaths.append(plot_single_parameter(fft_freq, np.abs(noise_params_nd['eta']), 'eta', '', "ND", date, DUT_DIR))
        plt.close()
    
    # calculate Rn (real impedance) for both
    Rn_cw = noise_params_cw['t']/(4*296.15)*R_0 # real impedance in Ohms
    Rn_nd = noise_params_nd['t']/(4*296.15)*R_0 # real impedance in Ohms
    
    if kwargs.get("graph_rn", True):
        filepaths.append(plot_single_parameter(fft_freq, Rn_cw, 'R_n', 'Ohms', "CW", date, DUT_DIR))
        filepaths.append(plot_single_parameter(fft_freq, Rn_nd, 'R_n', 'Ohms', "ND", date, DUT_DIR))

    if kwargs.get("graph_gamma_opt", True):
        filepaths.append(plot_complex_parameter(fft_freq, noise_params_cw['gamma_opt'], '\\Gamma_{opt}', 'dB', "CW", date, DUT_DIR))
        filepaths.append(plot_complex_parameter(fft_freq, noise_params_nd['gamma_opt'], '\\Gamma_{opt}', 'dB', "ND", date, DUT_DIR))
    
    if kwargs.get("graph_t_min", True):
        filepaths.append(plot_single_parameter(fft_freq, noise_params_cw['T_min'], 'T_{min}', 'K', "CW", date, DUT_DIR))
        filepaths.append(plot_single_parameter(fft_freq, noise_params_nd['T_min'], 'T_{min}', 'K', "ND", date, DUT_DIR))

    if kwargs.get("graph_te", True):
        filepaths.append(plot_single_parameter(fft_freq, noise_params_cw['Te'], 'T_e', 'K', "CW", date, DUT_DIR))
        filepaths.append(plot_single_parameter(fft_freq, noise_params_nd['Te'], 'T_e', 'K', "ND", date, DUT_DIR))
    
    # -------------------- Calculate and plot Noise Figure for both --------------------
    # Calculate noise figure for both
    nf_cw = 10 * np.log10((T_AMB + noise_params_cw['Te']) / T_AMB)
    nf_nd = 10 * np.log10((T_AMB + noise_params_nd['Te']) / T_AMB)

    if kwargs.get("graph_nf", True):
        filepaths.append(plot_single_parameter(fft_freq, nf_cw, 'NF', 'dB', "CW", date, DUT_DIR))
        filepaths.append(plot_single_parameter(fft_freq, nf_nd, 'NF', 'dB', "ND", date, DUT_DIR))
        filepaths.append(plot_parameter_comparison(fft_freq, nf_cw, nf_nd, 'NF', 'dB', date, DUT_DIR))
        
    # -------------------- Save results --------------------
    xparams_df = pd.DataFrame({
        'Frequency (GHz)': fft_freq / 1e9,
        'CW xn1_sq': xn1_sq_cw,
        'CW xn2_sq': xn2_sq_cw,
        'CW xn1_xn2_conj': xn1_xn2_conj_cw,
        'CW Te': noise_params_cw['Te'],
        'CW T_min': noise_params_cw['T_min'],
        'CW gamma_opt': noise_params_cw['gamma_opt'],
        'CW gamma_G': noise_params_cw['gamma_G'],
        'CW t': noise_params_cw['t'],
        'CW Rn': Rn_cw,
        'CW NF': nf_cw,
        'ND xn1_sq': xn1_sq_nd,
        'ND xn2_sq': xn2_sq_nd,
        'ND xn1_xn2_conj': xn1_xn2_conj_nd,
        'ND Te': noise_params_nd['Te'],
        'ND T_min': noise_params_nd['T_min'],
        'ND gamma_opt': noise_params_nd['gamma_opt'],
        'ND gamma_G': noise_params_nd['gamma_G'],
        'ND t': noise_params_nd['t'],
        'ND Rn': Rn_nd,
        'ND NF': nf_nd,
    })
    filepaths.append(os.path.join(DUT_DIR, f"Processed_DUT_{date}.csv"))
    xparams_df.to_csv(filepaths[-1], index=False)

    return filepaths

if __name__ == "__main__":
    dut_main()