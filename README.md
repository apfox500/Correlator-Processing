# Correlator Processing

This repository provides tools for processing and analyzing data from the correaltor instrument for measurment of low noise amplifiers and transistors. It includes scripts for processing continuous wave (CW) measurements, noise diode (ND) calibration, and device under test (DUT) measurements, as well as a command-line interface (CLI) for interactive use.

## Features

- Process CW, ND, and DUT data with dedicated scripts
- Interactive CLI for guided data processing
- Command-line flags for automation and scripting
- Output and data validation

## Environment Setup

1. **Clone the repository:**

   ```sh
   git clone https://github.com/apfox500/Correlator-Processing.git
   cd "Correlator Processing"
   ```

2. **Set up the Python environment:**
   - Recommended: Use the provided `environment.yml` file (requires [conda](https://docs.conda.io/en/latest/))

   ```sh
   cd Process
   conda env create -f environment.yml
   conda activate correlator-processing
   ```

   - Or, install requirements manually:

   ```sh
   pip install -r requirements.txt
   ```

## How to Run

All main processing scripts are in `Process/main.py`. Before running the code, edit the [`Process/constants.py`](Process/constants.py) file to set the correct directories, dates, and filenames. You can run the code in several ways:

### 1. Interactive CLI

The interactive CLI provides a menu-driven interface that allows you to:

- Process Continuous Wave (CW) data with optional S-parameter file selection and plotting.
- Process Noise Diode (ND) calibration data, including custom S-parameter file selection and sample count.
- Process Device Under Test (DUT) data with configurable gain file input.
- Exit the program safely.

The CLI guides you through each step, prompting for required parameters and offering defaults (sourced from [`Process/constants.py`](Process/constants.py)). To choose the default, simply press 'Enter'.

Launch the menu-driven CLI:

```sh
python Process/main.py --cli
```

### 2. Direct Processing Modes

You can run each processing mode directly with command-line flags:

- **CW Processing:**

  ```sh
  python Process/main.py --cw [--cw-date YYYY-MM-DD] [--cw-filename FILENAME] [--cw-graph 0|1|2]
  ```

- **Noise Diode Processing:**

  ```sh
  python Process/main.py --nd [--nd-date YYYY-MM-DD] [--nd-filename FILENAME] [--nd-num-samples N] [--nd-ch1-file PATH] [--nd-ch2-file PATH]
  ```

- **DUT Processing:**

  ```sh
  python Process/main.py --dut [--dut-date YYYY-MM-DD] [--dut-gain-file PATH]
  ```

### 3. Using VS Code

Pre-configured run/debug options are available in `.vscode/launch.json` (or `Process/launch.json`). Use the Run/Debug panel to select and launch CLI, CW, ND, or DUT modes.

## Output

Processed data and plots are saved in the `correlator_testing_output/` directory.

## Media

Supplementary media files (CAD images, Assembled images, presentation, sample graphs) related to the correlator instrument can be found in the `media/` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
