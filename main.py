import argparse
import os
import sys
from datetime import datetime

from Constants import *
from CW import CW_main
from DUT import dut_main
from LoadCal import loadcal_main
from ND import noise_main
from utils import print_fps

def create_run_summary(module_name, date, filepaths, **kwargs):
    """
    Create a summary text file documenting the processing run.
    
    Parameters:
        module_name (str): Name of the processing module (e.g., "CW", "ND", "LoadCal", "DUT")
        date (str): Date string used for the run (this is the date of the original data)
        filepaths (list): List of output file paths
        **kwargs: Additional parameters used in the run
    """
    # Determine output directory based on module
    output_dirs = {
        "CW": CW_DIR,
        "ND": ND_DIR, 
        "LoadCal": LOAD_DIR,
        "DUT": DUT_DIR
    }
    
    output_dir = output_dirs.get(module_name, OUT_DIR)
    summary_file = os.path.join(output_dir, f"{module_name}_Summary_{date}.txt")
    
    # Get current timestamp for when this summary was generated
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Data source information based on module
    data_sources = {
        "CW": {
            "source_dir": DATA_DIR,
            "data_pattern": CW_FILENAME,
            "s_param_files": [FILE1, FILE2, FILE3, FILE4, FILE5, FILE6],
            "example_filename": f"f_CW1.5GHz_PwrSensorOnCouplerZUDC20-5R23-S+PortB_Split02P1P2OnCorrelatorCh1Ch2_NSamp131072_PWR-30dBm_Att20dB_Ch115dBCh225dB__Fs10.0GHz_{date}_0.npy",
            "example_format": "Binary numpy array containing complex voltage samples from correlator channels"
        },
        "ND": {
            "source_dir": DATA_DIR,
            "data_pattern": ND_FILENAME,
            "s_param_files": [CAL_FILE, CH1_FILE, CH2_FILE],
            "example_filename": f"NoiseDiode136_AttCh115dBCh225dB_Splitter02_Port0_NSamp2097152_Att0dB__Fs10.0GHz_{date}_0.npy",
            "example_format": "Binary numpy array containing noise diode measurement samples"
        },
        "LoadCal": {
            "source_dir": DATA_DIR,
            "data_pattern": LOAD_FILENAME,
            "s_param_files": [],
            "example_filename": f"LoadCal_AttCh1_15dBCh2_25dB_NSamp2097152__Fs10.0GHz_{date}_0.npy",
            "example_format": "Binary numpy array containing 50-ohm load termination measurements"
        },
        "DUT": {
            "source_dir": DATA_DIR,
            "data_pattern": DUT_FILENAME,
            "s_param_files": [DUT_S_FILE],
            "example_filename": f"DUT_ZKL2Plus_AttCh1_15dBCh2_25dB_NSamp2097152__Fs10.0GHz_{date}_0.npy",
            "example_format": "Binary numpy array containing device under test measurements"
        }
    }
    
    current_source = data_sources.get(module_name, {})
    
    # CSV format examples
    csv_examples = {
        "CW": {
            "Processed_CW_": {
                "description": "Complex S-parameters in Python complex format",
                "example": "Frequency,S31,S46\\n1.0,(0.5+0.2j),(0.3-0.1j)\\n1.1,(0.52+0.18j),(0.31-0.08j)\\n..."
            }
        },
        "ND": {
            "Processed_Noise_": {
                "description": "Complex channel gains and phase differences",
                "example": "Freq (GHz),Ch1 Gain,Ch2 Gain,Phase Difference (rad)\\n1.0,(0.8+0.1j),(0.9-0.05j),0.15\\n1.1,(0.81+0.09j),(0.91-0.04j),0.14\\n..."
            }
        },
        "LoadCal": {
            "Processed_Load_": {
                "description": "Power spectral density values for calibration",
                "example": "Freq,Ch1 PSD,Ch2 PSD\\n1.0,1.2e-12,1.1e-12\\n1.1,1.3e-12,1.2e-12\\n..."
            }
        },
        "DUT": {
            "Processed_DUT_": {
                "description": "Complete noise parameter analysis with both CW and ND corrections",
                "example": "Frequency,Gain_CW,Gain_ND,NF_CW,NF_ND,Tmin_CW,Tmin_ND,GammaOpt_Real_CW,GammaOpt_Imag_CW\\n1.0,25.2,25.1,2.1,2.2,290,295,0.1,0.05\\n..."
            }
        }
    }
    
    # File descriptions based on common patterns
    file_descriptions = {
        # CW files
        "CW_Power_": "Output power vs frequency plot showing corrected power from CW measurement",
        "CW_Gain_Phase_": "Channel gains (dB) and phase difference (degrees) vs frequency", 
        "Processed_CW_": "Complex S-parameters (S31, S46) in CSV format for gain correction",
        
        # ND files
        "Noise_Gain_Phase_": "Average channel gains (dB) and phase difference (degrees) vs frequency",
        "Processed_Noise_": "Complex channel gains and phase difference in CSV format",
        
        # LoadCal files
        "Load_Cal_Temp_": "Load calibration temperature vs frequency",
        "Processed_Load_": "Load calibration data with CW and ND corrections in CSV format",
        
        # DUT files
        "DUT_XParameters_CW_": "X-parameters magnitude and phase vs frequency (CW gain)",
        "DUT_XParameters_ND_": "X-parameters magnitude and phase vs frequency (ND gain)", 
        "DUT_XParameters_Comparison_": "Comparison of X-parameters between CW and ND gain methods",
        "DUT_Eta_CW_": "Noise correlation parameter |eta| vs frequency (CW gain)",
        "DUT_Eta_ND_": "Noise correlation parameter |eta| vs frequency (ND gain)",
        "DUT_Rn_CW_": "Equivalent noise resistance vs frequency (CW gain)",
        "DUT_Rn_ND_": "Equivalent noise resistance vs frequency (ND gain)",
        "DUT_GammaOpt_CW_": "Optimal reflection coefficient magnitude and phase vs frequency (CW gain)",
        "DUT_GammaOpt_ND_": "Optimal reflection coefficient magnitude and phase vs frequency (ND gain)",
        "DUT_Tmin_CW_": "Minimum noise temperature vs frequency (CW gain)",
        "DUT_Tmin_ND_": "Minimum noise temperature vs frequency (ND gain)",
        "DUT_Te_CW_": "Effective noise temperature vs frequency (CW gain)",
        "DUT_Te_ND_": "Effective noise temperature vs frequency (ND gain)",
        "DUT_NF_CW_": "Noise figure vs frequency (CW gain)",
        "DUT_NF_ND_": "Noise figure vs frequency (ND gain)",
        "DUT_NF_Comparison_": "Comparison of noise figure between CW and ND gain methods",
        "Processed_DUT_": "Complete noise analysis results with both CW and ND data in CSV format"
    }
    
    with open(summary_file, 'w') as f:
        f.write(f"=== {module_name} Processing Summary ===\n")
        f.write(f"Summary Generated: {generation_time}\n")
        f.write(f"Data Date: {date}\n")
        f.write(f"Processing Script: {os.path.basename(__file__)}\n\n")
        
        f.write("=== Data Sources ===\n")
        f.write(f"Source Directory: {current_source.get('source_dir', 'N/A')}\n")
        f.write(f"Data Pattern: {current_source.get('data_pattern', 'N/A')}\n")
        f.write(f"Example Input File: {current_source.get('example_filename', 'N/A')}\n")
        f.write(f"Input Format: {current_source.get('example_format', 'N/A')}\n")
        
        if current_source.get('s_param_files'):
            f.write(f"\nS-Parameter/Calibration Files Used:\n")
            for s_file in current_source['s_param_files']:
                f.write(f"  - {s_file}\n")
        
        # Add dependency files for certain modules
        if module_name == "LoadCal":
            f.write(f"\nDependency Files:\n")
            if kwargs.get('gain_file'):
                f.write(f"  - CW Gain File: {kwargs['gain_file']}\n")
            if kwargs.get('nd_gain_file'):
                f.write(f"  - ND Gain File: {kwargs['nd_gain_file']}\n")
        elif module_name == "DUT":
            f.write(f"\nDependency Files:\n")
            if kwargs.get('cw_gain_file'):
                f.write(f"  - CW Gain File: {kwargs['cw_gain_file']}\n")
            if kwargs.get('nd_gain_file'):
                f.write(f"  - ND Gain File: {kwargs['nd_gain_file']}\n")
            if kwargs.get('load_file'):
                f.write(f"  - Load Cal File: {kwargs['load_file']}\n")
        
        f.write("\n=== Processing Constants ===\n")
        f.write(f"Number of Traces: {kwargs.get('num_traces', NUM_TRACES)}\n")
        f.write(f"Sample Cutoff: {SAMPLE_CUTOFF}\n")
        f.write(f"Dataset Length: {DATASET_LENGTH} (frequency bins)\n")
        f.write(f"Sampling Frequency: {FS} GHz\n")
        f.write(f"Frequency Range: 1-2 GHz\n")
        f.write(f"Characteristic Impedance: {R_0} Ohms\n")
        
        # Module-specific constants
        if module_name == "CW":
            f.write(f"Graph Flag: {kwargs.get('graph_flag', 0)}\n")
        elif module_name == "ND":
            f.write(f"CH1 S-parameter File: {kwargs.get('ch1_file', CH1_FILE)}\n")
            f.write(f"CH2 S-parameter File: {kwargs.get('ch2_file', CH2_FILE)}\n")
            f.write(f"Calibration File: {kwargs.get('cal_file', CAL_FILE)}\n")
        elif module_name == "DUT":
            f.write(f"CW Gain File: {kwargs.get('cw_gain_file', 'Default')}\n")
            f.write(f"ND Gain File: {kwargs.get('nd_gain_file', 'Default')}\n")
            f.write(f"Load File: {kwargs.get('load_file', LOAD_FILE)}\n")
            f.write(f"DUT S-parameter File: {kwargs.get('dut_s_file', DUT_S_FILE)}\n")
        
        f.write(f"\n=== Output Files ({len(filepaths)} total) ===\n")
        
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            file_ext = os.path.splitext(filename)[1]
            
            # Find matching description
            description = "Data file"
            for pattern, desc in file_descriptions.items():
                if pattern in filename:
                    description = desc
                    break
            
            # Add file type info
            if file_ext == '.png':
                description += " (Plot/Graph)"
            elif file_ext == '.csv':
                description += " (Data)"
            
            f.write(f"{filename}\n")
            f.write(f"  Path: {filepath}\n")
            f.write(f"  Description: {description}\n")
            
            # Add example format for CSV files
            if file_ext == '.csv':
                for pattern, example_info in csv_examples.get(module_name, {}).items():
                    if pattern in filename:
                        if isinstance(example_info, dict):
                            f.write(f"  Format: {example_info['description']}\n")
                            f.write(f"  Example Content:\n    {example_info['example'] if 'example' in example_info else 'See description above'}\n")
                        else:
                            f.write(f"  Example Content:\n    {example_info}\n")
                        break
            f.write("\n")
        
        f.write("=== Processing Notes ===\n")
        if module_name == "CW":
            f.write("- CW processing measures voltage from FFT and calculates complex S-parameters\n")
            f.write("- Alpha/beta corrections applied if S-parameter files available\n")
            f.write("- Output S-parameters represent voltage gain (complex)\n")
        elif module_name == "ND":
            f.write("- ND processing measures power from PSD and normalizes by noise diode reference\n") 
            f.write("- Converts power gains to voltage-equivalent gains for consistency with CW\n")
            f.write("- S-parameter corrections applied using complex transfer functions\n")
        elif module_name == "LoadCal":
            f.write("- Load calibration using 50-ohm termination measurements\n")
            f.write("- Provides correction factors for both CW and ND gain methods\n")
        elif module_name == "DUT":
            f.write("- Dual analysis using both CW and ND gain corrections\n")
            f.write("- Calculates complete noise parameters including NF, T_min, gamma_opt\n")
            f.write("- Generates comparison plots between CW and ND methods\n")
    
    print(f"Summary saved to: {summary_file}")
    return summary_file

# Functions to run each process with default parameters
def run_load_cal():
    """Run Load Cal processing with default parameters."""
    print("Running Load Cal processing...")
    filepaths = loadcal_main(graph=False)
    print_fps(filepaths)
    
    # Create summary file
    summary_file = create_run_summary("LoadCal", LOAD_DATE, filepaths)
    print("Load Cal processing completed.\n")

def run_cw():
    """Run CW processing with default parameters."""
    print("Running CW processing...")
    filepaths = CW_main()
    print_fps(filepaths)
    
    # Create summary file
    summary_file = create_run_summary("CW", CW_DATE, filepaths)
    print("CW processing completed.\n")

def run_noise():
    """Run Noise Diode processing with default parameters."""
    print("Running Noise Diode processing...")
    filepaths = noise_main()
    print_fps(filepaths)
    
    # Create summary file
    summary_file = create_run_summary("ND", ND_DATE, filepaths)
    print("Noise Diode processing completed.\n")

def run_dut():
    """Run DUT processing with default parameters."""
    print("Running DUT processing...")
    filepaths = dut_main()
    print_fps(filepaths)
    
    # Create summary file
    summary_file = create_run_summary("DUT", DUT_DATE, filepaths)
    print("DUT processing completed.\n")

def run_all(num_traces: int):
    """Run all processing steps in order: CW, Noise Diode, Load Cal, DUT."""
    print("Running all processing steps...\n")
    
    # Run CW and capture the output CSV file
    print("Running CW processing...")
    cw_filepaths = CW_main()
    print_fps(cw_filepaths)
    create_run_summary("CW", CW_DATE, cw_filepaths)
    print("CW processing completed.\n")
    
    # Find the CW Complex Gain CSV file from the outputs
    cw_gain_csv_file = None
    for filepath in cw_filepaths:
        if filepath.endswith('.csv') and 'CW_Complex_Gain_' in filepath:
            cw_gain_csv_file = filepath
            break
    
    # Run Noise Diode and capture the output CSV file
    print("Running Noise Diode processing...")
    noise_filepaths = noise_main(date=ND_DATE, num_traces=num_traces)
    print_fps(noise_filepaths)
    create_run_summary("ND", ND_DATE, noise_filepaths)
    print("Noise Diode processing completed.\n")
    
    # Find the ND Gain CSV file from the outputs
    nd_gain_csv_file = None
    for filepath in noise_filepaths:
        if filepath.endswith('.csv') and 'Processed_Noise_' in filepath:
            nd_gain_csv_file = filepath
            break
    
    # Run Load Cal with both CW and ND gain files
    print("Running Load Cal processing...")
    load_kwargs = {'graph': False, 'date': LOAD_DATE, 'num_traces': num_traces}
    if cw_gain_csv_file:
        load_kwargs['gain_file'] = cw_gain_csv_file
        print(f"Using CW gain file: {cw_gain_csv_file}")
    if nd_gain_csv_file:
        load_kwargs['nd_gain_file'] = nd_gain_csv_file
        print(f"Using ND gain file: {nd_gain_csv_file}")
    
    load_filepaths = loadcal_main(**load_kwargs)
    print_fps(load_filepaths)
    create_run_summary("LoadCal", filepaths=load_filepaths, **load_kwargs)
    print("Load Cal processing completed.\n")
    
    # Find the LoadCal CSV file from the outputs
    load_csv_file = None
    for filepath in load_filepaths:
        if filepath.endswith('.csv') and 'Processed_Load_' in filepath:
            load_csv_file = filepath
            break
    # Run DUT with the CSV files from previous steps
    print("Running DUT processing...")
    dut_kwargs = {'date': DUT_DATE, 'num_traces': num_traces}
    if cw_gain_csv_file:
        dut_kwargs['cw_gain_file'] = cw_gain_csv_file
        print(f"Using CW gain file: {cw_gain_csv_file}")
    if nd_gain_csv_file:
        dut_kwargs['nd_gain_file'] = nd_gain_csv_file
        print(f"Using ND gain file: {nd_gain_csv_file}")
    if load_csv_file:
        dut_kwargs['load_file'] = load_csv_file
        print(f"Using load file: {load_csv_file}")
    
    dut_filepaths = dut_main(**dut_kwargs)
    print_fps(dut_filepaths)
    create_run_summary("DUT", filepaths=dut_filepaths, **dut_kwargs)
    print("DUT processing completed.\n")
    
    print("All processing completed!")

def cli_interface():
    """
    Command-line interface for processing data.
    """
    def print_menu():
        print("\nCorrelator Processing CLI")
        print("1. Process CW")
        print("2. Process Noise Diode")
        print("3. Process Load Cal")
        print("4. Process DUT")
        print("5. Process All (30 traces)")
        print("6. Process All (500 traces)")
        print("7. Exit")

    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print_menu()
        choice = input("Select an option (1-6): ").strip()
        
        if choice == "1":  # Process CW
            run_cw()
            input("Press Enter to continue...")
        elif choice == "2":  # Process Noise Diode
            run_noise()
            input("Press Enter to continue...")
        elif choice == "3":  # Process Load Cal
            run_load_cal()
            input("Press Enter to continue...")
        elif choice == "4":  # Process DUT
            run_dut()
            input("Press Enter to continue...")
        elif choice == "5":  # Process All (30)
            run_all(num_traces=30)
            input("Press Enter to continue...")
        elif choice == "6":  # Process All (500 traces)
            run_all(num_traces=500)
            input("Press Enter to continue...")
        elif choice == "7":  # Exit
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid option. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlator Processing Main Entry Point")
    parser.add_argument('--cli', action='store_true', help='Run interactive CLI')
    parser.add_argument('--all', action='store_true', help='Run all processing steps (Load Cal, CW, Noise, DUT)')
    
    # Load Cal flags
    parser.add_argument('--load', action='store_true', help='Run loadcal_main (Load Cal) directly')
    parser.add_argument('--load-date', type=str, default=LOAD_DATE, help='Date string for Load Cal data')
    parser.add_argument('--load-filename', type=str, default=LOAD_FILENAME, help='Base filename for Load Cal data')
    parser.add_argument('--load-num-samples', type=int, default=500, help='Number of samples to test for Load Cal')
    parser.add_argument('--load-graph', action='store_true', help='Show PSD and Phase Difference Graph for Load Cal')
    
    # CW flags
    parser.add_argument('--cw', action='store_true', help='Run CW_main directly')
    parser.add_argument('--cw-date', type=str, default=CW_DATE, help='Date string for CW data')
    parser.add_argument('--cw-filename', type=str, default=CW_FILENAME, help='Filename for CW data')
    parser.add_argument('--cw-graph', type=int, default=0, help='Graph frequency response flag (0/1/2)')
    
    # ND flags
    parser.add_argument('--nd', action='store_true', help='Run noise_main (Noise Diode) directly')
    parser.add_argument('--nd-date', type=str, default=ND_DATE, help='Date string for Noise Diode data')
    parser.add_argument('--nd-filename', type=str, default=ND_FILENAME, help='Base filename for Noise Diode data')
    parser.add_argument('--nd-num-samples', type=int, default=500, help='Number of samples to test for Noise Diode')
    parser.add_argument('--nd-ch1-file', type=str, default=CH1_FILE, help='S-parameter file for CH1 (Noise Diode)')
    parser.add_argument('--nd-ch2-file', type=str, default=CH2_FILE, help='S-parameter file for CH2 (Noise Diode)')
    
    # DUT flags
    parser.add_argument('--dut', action='store_true', help='Run dut_main (DUT) directly')
    parser.add_argument('--dut-date', type=str, default=DUT_DATE, help='Date string for DUT data')
    parser.add_argument('--dut-filename', type=str, default=DUT_FILENAME, help='Base filename for DUT data')
    parser.add_argument('--dut-gain-file', type=str, default=CW_GAIN_FILE, help='Gain file path for DUT')
    parser.add_argument('--dut-load-file', type=str, default=LOAD_FILE, help='Load file path for DUT')
    parser.add_argument('--dut-num-samples', type=int, default=NUM_TRACES, help='Number of samples to test for DUT')
    
    args = parser.parse_args()

    # Ensure output directory exists
    if OUT_DIR and not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Ensure data directory exists
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist or is empty.")

    if args.all:
        print("Running all processing steps...")
        run_all()
    elif args.load:
        print(f"Running loadcal_main with date={args.load_date}, filename={args.load_filename}, num_traces={args.load_num_samples}")
        loadcal_main(
            date=args.load_date,
            filename=args.load_filename,
            num_traces=args.load_num_samples,
            graph=args.load_graph
        )
    elif args.cw:
        print(f"Running CW_main with date={args.cw_date}, filename={args.cw_filename}, graph_flag={args.cw_graph}")
        CW_main(date=args.cw_date, filename=args.cw_filename, graph_flag=args.cw_graph, file_kwargs={})
    elif args.nd:
        print(f"Running noise_main with date={args.nd_date}, filename={args.nd_filename}, num_traces={args.nd_num_samples}")
        noise_main(
            date=args.nd_date,
            filename=args.nd_filename,
            num_traces=args.nd_num_samples,
            ch1_file=args.nd_ch1_file,
            ch2_file=args.nd_ch2_file
        )
    elif args.dut:
        print(f"Running dut_main with date={args.dut_date}, filename={args.dut_filename}")
        dut_main(
            date=args.dut_date,
            filename=args.dut_filename,
            gain_file=args.dut_gain_file,
            gain_headers=CW_GAIN_HEADERS,
            load_file=args.dut_load_file,
            load_headers=LOAD_HEADERS,
            num_traces=args.dut_num_samples
        )
    elif args.cli or not (args.all or args.load or args.cw or args.nd or args.dut):
        cli_interface()