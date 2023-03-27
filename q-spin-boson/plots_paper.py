from dotenv import load_dotenv  # load environment variables from env file
import os
import numpy as np

from settings.types import Env, Model
from src.model_spinboson import SSpinBosonSimulation, DSpinBosonSimulation
from src.model_twolevel import TwoLvlSimulation
from src.model_js import JCSimulation
import src.plotting as plotting

def main():
    print('-'*80)

    sim = SSpinBosonSimulation()
    print(sim)
    sim.check_results()
    sim = DSpinBosonSimulation()
    print(sim)
    sim.check_results()
    sim = TwoLvlSimulation()
    print(sim)
    sim.check_results()
    sim = SSpinBosonSimulation(model=Model.SB1SPZ)
    print(sim)
    sim.check_results()
    sim = SSpinBosonSimulation(model=Model.SB1SJC)
    print(sim)
    sim.check_results()
    sim = JCSimulation()
    sim.check_results()

    fig = plotting.plot_states(sim, exact=True)
    fig.show()

    fig = plotting.plot_ifid_vs_dt_env()
    fig.show()

    fig = plotting.plot_ifid_vs_dt_noises()
    fig.show()

    fig = plotting.plot_ifid_vs_noise()
    fig.show()

    fig = plotting.plot_ifid_vs_time()
    fig.show()

    fig = plotting.plot_ifid_vs_gamma()
    fig.show()

    fig = plotting.plot_ifid_vs_time_gammas()
    fig.show()

    fig = plotting.plot_bosons()
    fig.show()

    fig = plotting.plot_spin()
    fig.show()

    # fig = plotting.plot_spincorrelation()
    # fig.show()

if __name__ == '__main__':
    main()