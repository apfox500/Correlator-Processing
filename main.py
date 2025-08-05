
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf
from tqdm import tqdm
import sys

from Constants import *
from CW import CW_main
from ND import noise_main
from DUT import dut_main

def cli_interface():
    """
    Command-line interface for processing data.
    """
    def print_menu():
        print("\nCorrelator Processing CLI")
        print("1. Process CW Data")
        print("2. Process Noise Diode Data")
        print("3. Process DUT Data")
        print("4. Exit")

    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print_menu()
        choice = input("Select an option (1-4): ").strip()
        if choice == "1": # Process CW Data
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

            if filepaths is not None:
                print("Data saved to:")
                for fp in filepaths:
                    print(fp)

            input("Press Enter to continue...")
        elif choice == "2": # Process Noise Diode Data
            date = input("Enter date string for Noise Diode data (default: {}): ".format(DATE)).strip() or DATE
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

            noise_main(date=date, data_file_base=file_base, num_samples_test=num_samples_test, **file_kwargs)
            input("Press Enter to continue...")
        elif choice == "3": # Process DUT Data
            date = input("Enter date string for DUT data (default: {}): ".format(DATE)).strip() or DATE
            gain_file = input("Enter gain file path (default: ./correlator_testing_output/Processed gain.csv): ").strip() or "./correlator_testing_output/Processed gain.csv"
            dut_main(date=date, gain_file=gain_file)
            input("Press Enter to continue...")
        elif choice == "4": # Exit
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
    parser.add_argument('--nd-date', type=str, default=DATE, help='Date string for Noise Diode data')
    parser.add_argument('--nd-filename', type=str, default=ND_FILENAME, help='Base filename for Noise Diode data')
    parser.add_argument('--nd-num-samples', type=int, default=500, help='Number of samples to test for Noise Diode')
    parser.add_argument('--nd-ch1-file', type=str, default=CH1_FILE, help='S-parameter file for CH1 (Noise Diode)')
    parser.add_argument('--nd-ch2-file', type=str, default=CH2_FILE, help='S-parameter file for CH2 (Noise Diode)')
    # DUT flags
    parser.add_argument('--dut', action='store_true', help='Run dut_main (DUT) directly')
    parser.add_argument('--dut-date', type=str, default=DATE, help='Date string for DUT data')
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

