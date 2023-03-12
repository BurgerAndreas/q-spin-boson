from dotenv import load_dotenv  # load environment variables from env file
import os
import numpy as np
from settings.conventions import test_convention

from src.model_spin_boson import SpinBosonSimulation

def main():
    print('-'*80)

    sim = SpinBosonSimulation()
    sim.get_simulation()
    sim.check_results()
    sim.print_results()

if __name__ == '__main__':
    main()