from dotenv import load_dotenv  # load environment variables from env file
import os
import numpy as np

from settings.types import Env
from settings.conventions import test_convention
from src.model_spinboson import SSpinBosonSimulation, DSpinBosonSimulation

def main():
    print('-'*80)

    sim = SSpinBosonSimulation()
    print(sim.name)
    sim.get_simulation()
    sim.check_results()
    # sim.print_results()

    sim.get_gates()
    sim.save_layout(sim.backend)
    sim.save_circuit_latex(sim.backend)
    sim.save_circuit_image(sim.backend)


if __name__ == '__main__':
    main()