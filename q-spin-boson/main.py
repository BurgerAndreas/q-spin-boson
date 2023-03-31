from dotenv import load_dotenv  # load environment variables from env file
import os
import numpy as np

from settings.types import Env
from settings.conventions import test_convention
from src.model_general import simulation
from src.model_spinboson import SSpinBosonSimulation, DSpinBosonSimulation
from src.model_jc import JCSimulation
from src.model_twolevel import TwoLvlSimulation

def main():
    print('-'*80)

    sim = SSpinBosonSimulation()
    print(sim.name)
    sim.get_simulation()
    sim.check_results()

    print(sim.infidelity)

    sim.get_gates()
    sim.save_layout()
    sim.save_circuit_latex()
    sim.save_circuit_image()


if __name__ == '__main__':
    main()