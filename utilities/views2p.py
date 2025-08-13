# views2p.py
#
# A script to read a Touchstone file (e.g., .s2p) and plot the
# magnitude of the S-parameters in dB versus frequency.


import skrf as rf
import matplotlib.pyplot as plt

def plot_s_parameters(filepath):
    """
    Reads an S-parameter file and plots the magnitude of all parameters.

    Args:
        filepath (str): The path to the Touchstone file.
    """
    try:
        # Load the Touchstone file using scikit-rf
        # The Network object contains all the S-parameter data
        network = rf.Network(filepath)

        # Create a figure and axes for the plot
        # We'll create a 2x2 grid for a 2-port network (S11, S21, S12, S22)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'S-Parameters for {network.name}', fontsize=16)

        # Plot S11
        network.plot_s_db(m=0, n=0, ax=axs[0, 0])
        axs[0, 0].set_title('S11 (Return Loss)')
        axs[0, 0].grid(True)

        # Plot S21
        network.plot_s_db(m=1, n=0, ax=axs[0, 1])
        axs[0, 1].set_title('S21 (Insertion Loss)')
        axs[0, 1].grid(True)

        # Plot S12
        network.plot_s_db(m=0, n=1, ax=axs[1, 0])
        axs[1, 0].set_title('S12 (Reverse Isolation)')
        axs[1, 0].grid(True)

        # Plot S22
        network.plot_s_db(m=1, n=1, ax=axs[1, 1])
        axs[1, 1].set_title('S22 (Output Return Loss)')
        axs[1, 1].grid(True)

        # Improve layout and show the plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':

    touchstone_filepath = 'alphabeta/DUT1.s2p'

    # Create a dummy s2p file for demonstration if it doesn't exist
    try:
        with open(touchstone_filepath, 'x') as f:
            f.write("! Example S2P File\n")
            f.write("# HZ S DB R 50\n")
            f.write("1e9 -10 0 -3 90 -10 0\n")
            f.write("2e9 -8 5 -4 85 -9 2\n")
            f.write("3e9 -6 10 -5 80 -8 4\n")
    except FileExistsError:
        # file already exists, do nothing
        pass

    # Call the function to plot the data
    plot_s_parameters(touchstone_filepath)
