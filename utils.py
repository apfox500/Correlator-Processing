import os

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from typing import Annotated
from typing import Union
from numpy.typing import NDArray


from Constants import *


def process_noise_data(data_file_base:str, num_samples_test:int=NUM_SAMPLES_TEST, **kwargs) -> tuple[
    Annotated[NDArray[np.complex128], (3001,)],
    Annotated[NDArray[np.complex128], (3001,)],
    Annotated[NDArray[np.complex128], (3001,)]
]:
    """
    Process all noise data files and return average PSD for ch1 and ch2.
    Parameters:
        data_file_base: The base string for the noise data filepaths (without index and .npy).
        num_samples_test: Number of samples/files to process.
        **kwargs: Additional keyword arguments for processing options.
    Returns:
        ch1_avg_psd, ch2_avg_psd: Average PSD arrays for ch1 and ch2 in V^2/Hz.
        csd_avg: Average Cross-Spectral Density (CSD) array.
    """
    # -------------------- Initialization --------------------
    fft_freq = kwargs.get("fft_freq", np.fft.rfftfreq(SAMPLE_CUTOFF, d=1/(FS*1e9)))
    power_data = np.zeros((3, 3001), dtype=np.complex128)
    

    # -------------------- Process Noise Files --------------------
    for idx in tqdm(range(num_samples_test), desc="Processing Noise Data", unit="file", total=num_samples_test):
        filename = f"{data_file_base}_{idx}.npy"
        if not os.path.exists(os.path.join(DATA_DIR, filename)):
            print(f"File {filename} does not exist in {DATA_DIR}. Skipping...")
            continue
        data_array = np.load(os.path.join(DATA_DIR, filename))
        ch1_psd, ch2_psd, csd = process_single_noise_data(data_array, fft_freq)
        power_data[0, :] += ch1_psd
        power_data[1, :] += ch2_psd
        power_data[2, :] += csd


    # -------------------- Calculate Average PSD --------------------
    ch1_avg_psd = power_data[0, :] / num_samples_test
    ch2_avg_psd = power_data[1, :] / num_samples_test
    csd_avg = power_data[2, :] / num_samples_test

    return ch1_avg_psd, ch2_avg_psd, csd_avg

def process_single_noise_data(data_array: NDArray, fft_freq: NDArray, **kwargs) -> tuple[NDArray, NDArray, NDArray]:
    """
    Processes noise data from two channels, computes their FFT, cross-spectral density (CSD), and power spectral density (PSD).

    Parameters:
        data_array (np.ndarray): 2D array of raw noise data with shape (2, N), where each row corresponds to a channel.
        fft_freq (np.ndarray): 1D array of frequency bins corresponding to the FFT output.
        **kwargs:
            graph_fft (bool, optional): If True, plots the FFT magnitude of both channels. Default is False.
            graph_psd (bool, optional): If True, plots the PSD of both channels. Default is False.

    Returns:
        tuple:
            ch1_psd (np.ndarray): Power spectral density of channel 1 (trimmed to 1-2 GHz).
            ch2_psd (np.ndarray): Power spectral density of channel 2 (trimmed to 1-2 GHz).
            csd (np.ndarray): Cross-spectral density (complex) between channel 1 and channel 2 (trimmed to 1-2 GHz).

    Notes:
        - Only the frequency range 1-2 GHz is considered for PSD and CSD calculations.
        - Plots are displayed if the corresponding keyword arguments are set to True.
    """

    graph_fft = kwargs.get("graph_fft", False)
    graph_psd = kwargs.get("graph_psd", False)

    # -------------------- Extract and Convert Data --------------------
    ch1_noise = data_array[0, :SAMPLE_CUTOFF]
    ch2_noise = data_array[1, :SAMPLE_CUTOFF]
    ch1_voltage = ch1_noise * VOLT_PER_TICK
    ch2_voltage = ch2_noise * VOLT_PER_TICK

    # -------------------- FFT and PSD Calculation --------------------
    ch1_fft = np.fft.rfft(ch1_voltage)
    ch2_fft = np.fft.rfft(ch2_voltage)

    if graph_fft:
        plt.figure(figsize=(10, 5))
        plt.plot(fft_freq, np.abs(ch1_fft), label='Ch1 FFT', color=COLORS[0])
        plt.plot(fft_freq, np.abs(ch2_fft), label='Ch2 FFT', color=COLORS[1])
        plt.title("FFT of Noise Data")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Trim FFT to 1-2 GHz
    mask = (fft_freq >= 1e9) & (fft_freq <= 2e9)
    ch1_fft = ch1_fft[mask]
    ch2_fft = ch2_fft[mask]

    csd = ch1_fft * np.conjugate(ch2_fft) # phase diff

    # Calculate PSD
    ch1_psd = 2 * np.abs(ch1_fft) ** 2 / (SAMPLE_CUTOFF * FS * 1e9)
    ch2_psd = 2 * np.abs(ch2_fft) ** 2 / (SAMPLE_CUTOFF * FS * 1e9)

    if graph_psd:
        plt.figure(figsize=(10, 5))
        plt.plot(fft_freq[mask] / 1e9, dB(ch1_psd), label='Ch1 PSD', color=COLORS[0])
        plt.plot(fft_freq[mask] / 1e9, dB(ch2_psd), label='Ch2 PSD', color=COLORS[1])
        plt.title("Power Spectral Density of Noise Data")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("PSD (dBm/Hz)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ch1_psd, ch2_psd, csd

def dB(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Convert a linear value or array to decibels (dB).
    Parameters:
        value (float or np.ndarray): Linear value(s) to convert. Must be > 0 for valid dB result.
    Returns:
        float or np.ndarray: Value(s) in dB. Returns -np.inf for elements <= 0.
    """
    return 10 * np.log10(value) if np.all(value > 0) else np.where(value > 0, 10 * np.log10(value), -np.inf)

def db(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Alias for dB function to maintain compatibility with existing code.
    Parameters:
        value (float or np.ndarray): Linear value(s) to convert.
    Returns:
        float or np.ndarray: Value(s) in dB.
    """
    return dB(value)

def dBm(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Convert a linear value or array (in Watts) to decibels-milliwatts (dBm).
    Parameters:
        value (float or np.ndarray): Linear value(s) in Watts. Must be > 0 for valid dBm result.
    Returns:
        float or np.ndarray: Value(s) in dBm. Returns -np.inf for elements <= 0.
    """
    return 10 * np.log10(value * 1e3) if np.all(value > 0) else np.where(value > 0, 10 * np.log10(value * 1e3), -np.inf)

def dbm(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Alias for dBm function to maintain compatibility with existing code.
    Parameters:
        value (float or np.ndarray): Linear value(s) in Watts.
    Returns:
        float or np.ndarray: Value(s) in dBm.
    """
    return dBm(value)

def linear(value: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Convert a value or array in decibels (dB) to linear scale.
    Parameters:
        value (float or np.ndarray): Value(s) in dB.
    Returns:
        float or np.ndarray: Linear value(s). Returns 0 for elements that are -np.inf.
    """
    return 10 ** (value / 10) if np.all(value > -np.inf) else np.where(value > -np.inf, 10 ** (value / 10), 0)

def print_fps(filepaths):
    if filepaths is not None:
        print("Data saved to:")
        for fp in filepaths:
            print(fp)

def parse_complex(s):
    if pd.isna(s) or s == '':
        return np.nan
    return complex(s.strip('()'))

def interp_complex(target:NDArray, source:NDArray, y:NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Interpolates a complex array y from source to target.
    Parameters:
        target (np.ndarray): Target x-coordinates.
        source (np.ndarray): Source x-coordinates.
        y (np.ndarray[np.complex128]): Complex values to interpolate.
    Returns:
        np.ndarray[np.complex128]: Interpolated complex values at target.
    """
    # interpolate phase and magnitude separately
    phase_interp = np.interp(target, source, np.angle(y))
    mag_interp = np.interp(target, source, np.abs(y))
    
    # combine magnitude and phase into complex numbers
    return mag_interp * np.exp(1j * phase_interp)

