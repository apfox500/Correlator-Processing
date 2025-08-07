import argparse
import os
import sys

from Constants import *
from CW import CW_main
from ND import noise_main
from DUT import dut_main
from LoadCal import loadcal_main
from utils import print_fps

def CW_cli():
    date = input("Enter date string for CW data (default: {}): ".format(CW_DATE)).strip() or CW_DATE
    filename = input("Enter filename for CW data (format: 'f_CW{freq}GHz_...__Fs{FS}GHz_{date}_0.npy'): ").strip() or CW_FILENAME
    graph_flag = int(input("Graph frequency response? (0 for No, 1 for first and last, 2 for all): ").strip()or '0')

    # 6 files for alpha/beta
    # Ask user if they want to change the Alpha/Beta S-parameter files
    change_files = input(
        "Do you want to change the Alpha/Beta S-parameter files? (y/n): "
    ).strip().lower()

    file_kwargs = {}
    if change_files == "y":
        print("\nAlpha/Beta S-parameter files:")
        print("1. FILE1 Contains SSP1A, SSP11 (default: {})".format(FILE1))
        file1 = input("Enter path for FILE1 (or press Enter to use default): ").strip() or FILE1
        file_kwargs["file1"] = file1

        print("2. FILE2 Contains SSPBB, SSPBA (default: {})".format(FILE2))
        file2 = input("Enter path for FILE2 (or press Enter to use default): ").strip() or FILE2
        file_kwargs["file2"] = file2

        print("3. FILE3 Contains GPM (default: {})".format(FILE3))
        file3 = input("Enter path for FILE3 (or press Enter to use default): ").strip() or FILE3
        file_kwargs["file3"] = file3

        print("4. FILE4 Contains GNCin (default: {})".format(FILE4))
        file4 = input("Enter path for FILE4 (or press Enter to use default): ").strip() or FILE4
        file_kwargs["file4"] = file4

        print("5. FILE5 Contains SSP66, SSP6A (default: {})".format(FILE5))
        file5 = input("Enter path for FILE5 (or press Enter to use default): ").strip() or FILE5
        file_kwargs["file5"] = file5

        print("6. FILE6 Contains GNCout (default: {})".format(FILE6))
        file6 = input("Enter path for FILE6 (or press Enter to use default): ").strip() or FILE6
        file_kwargs["file6"] = file6

    filepaths = CW_main(date=date, filename=filename, graph_flag=graph_flag, file_kwargs=file_kwargs)

    print_fps(filepaths)
    input("Press Enter to continue...")

def ND_cli():
    date = input("Enter date string for Noise Diode data (default: {}): ".format(ND_DATE)).strip() or ND_DATE
    file_base = input("Enter base filename for Noise Diode data (format: 'NoiseDiode..._Fs10.0GHz_{date}'): ").strip() or ND_FILENAME
    num_samples_test = int(input("Enter number of samples to test (default: 500): ").strip() or '500')

    change_files = input(
        "Do you want to change the S-parameter files for Noise Diode? (y/n): "
    ).strip().lower()

    file_kwargs = {}
    if change_files == "y":
        print("\nS-parameter files:")
        print("1. CH1_FILE (default: {})".format(CH1_FILE))
        ch1_file = input("Enter path for CH1_FILE (or press Enter to use default): ").strip() or CH1_FILE
        file_kwargs["ch1_file"] = ch1_file

        print("2. CH2_FILE (default: {})".format(CH2_FILE))
        ch2_file = input("Enter path for CH2_FILE (or press Enter to use default): ").strip() or CH2_FILE
        file_kwargs["ch2_file"] = ch2_file

    filepaths = noise_main(date=date, data_file_base=file_base, num_samples_test=num_samples_test, **file_kwargs)

    print_fps(filepaths)
    
    input("Press Enter to continue...")

def LoadCal_cli():
    date = input("Enter date string for Load Cal data (default: {}): ".format(LOAD_DATE)).strip() or LOAD_DATE
    file_base = input("Enter base filename for Load Cal data (format: 'LoadCal..._Fs10.0GHz_{date}'): ").strip() or LOAD_FILENAME
    num_samples_test = int(input("Enter number of samples to test (default: 500): ").strip() or '500')
    graph = input("Show PSD and Phase Difference Graph? (y/n): ").strip().lower() or "n"
    filepaths = loadcal_main(
        date=date,
        data_file_base=file_base,
        num_samples_test=num_samples_test,
        graph=(graph=="y")
    )

    print_fps(filepaths)
    input("Press Enter to continue...")



def DUT_cli():
    date = input("Enter date string for DUT data (default: {}): ".format(DUT_DATE)).strip() or DUT_DATE
    file_base = input("Enter base filename for DUT data (format: 'DUT_..._Fs10.0GHz_{date}'): ").strip() or DUT_FILENAME
    
    # Gain info
    gain_file = input("Enter complex gain file path (default: {}): ".format(GAIN_FILE)).strip() or GAIN_FILE
    gain_headers_input = input(f"Enter headers for [frequency, S31, S46] as comma-separated values (default: {', '.join(GAIN_HEADERS)}): ").strip()
    gain_headers = [h.strip() for h in gain_headers_input.split(",")] if gain_headers_input else GAIN_HEADERS

    # Load Cal info
    load_file = input("Enter load file path (default: {}): ".format(LOAD_FILE)).strip() or LOAD_FILE
    load_headers_input = input(f"Enter headers for [frequency, ch1 noise, ch2 noise, phase] as comma-separated values (default: {', '.join(LOAD_HEADERS)}): ").strip()
    load_headers = [h.strip() for h in load_headers_input.split(",")] if load_headers_input else LOAD_HEADERS

    num_samples_test = int(input("Enter number of samples to test (default: {}): ".format(NUM_SAMPLES_TEST)).strip() or f'{NUM_SAMPLES_TEST}')


    filepaths = dut_main(
        date=date, 
        data_file_base=file_base, 
        gain_file=gain_file, 
        gain_headers=gain_headers,
        load_file=load_file,
        load_headers=load_headers,
        num_samples_test=num_samples_test
    )

    print_fps(filepaths)
    input("Press Enter to continue...")

def cli_interface():
    """
    Command-line interface for processing data.
    """
    def print_menu():
        print("\nCorrelator Processing CLI")
        print("1. Process CW Data")
        print("2. Process Noise Diode Data")
        print("3. Process Load Cal Data")
        print("4. Process DUT Data")
        print("5. Exit")

    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print_menu()
        choice = input("Select an option (1-4): ").strip()
        if choice == "1": # Process CW Data
            CW_cli()
        elif choice == "2": # Process Noise Diode Data
            ND_cli()
        elif choice == "3": # Process Load Cal Data
            LoadCal_cli()
        elif choice == "4": # Process DUT Data
            DUT_cli()
        elif choice == "5": # Exit
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid option. Please try again.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlator Processing Main Entry Point")
    parser.add_argument('--cli', action='store_true', help='Run interactive CLI')
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
    parser.add_argument('--dut-gain-file', type=str, default="./correlator_testing_output/Processed gain.csv", help='Gain file path for DUT')
    args = parser.parse_args()

    # Ensure output directory exists
    if OUT_DIR and not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Ensure data directory exists
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' does not exist or is empty.")

    if args.cw:
        print(f"Running CW_main with date={args.cw_date}, filename={args.cw_filename}, graph_flag={args.cw_graph}")
        CW_main(date=args.cw_date, filename=args.cw_filename, graph_flag=args.cw_graph, file_kwargs={})
    elif args.nd:
        print(f"Running noise_main with date={args.nd_date}, data_file_base={args.nd_filename}, num_samples_test={args.nd_num_samples}")
        noise_main(
            date=args.nd_date,
            data_file_base=args.nd_filename,
            num_samples_test=args.nd_num_samples,
            ch1_file=args.nd_ch1_file,
            ch2_file=args.nd_ch2_file
        )
    elif args.dut:
        print(f"Running dut_main with date={args.dut_date}, gain_file={args.dut_gain_file}")
        dut_main(date=args.dut_date, gain_file=args.dut_gain_file)
    elif args.cli or not (args.cw or args.nd or args.dut):
        cli_interface()