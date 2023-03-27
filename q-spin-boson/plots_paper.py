from dotenv import load_dotenv  # load environment variables from env file
import os
import numpy as np

from settings.conventions import test_convention
from settings.types import Env, Model, H, Enc
from src.model_spinboson import SSpinBosonSimulation, DSpinBosonSimulation
from src.model_twolevel import TwoLvlSimulation
from src.model_js import JCSimulation
import src.plotting as plotting


def test_models():
    print('-'*80)

    sim = TwoLvlSimulation()
    sim = SSpinBosonSimulation(n_bos=4)
    sim = DSpinBosonSimulation(n_bos=4)
    sim = SSpinBosonSimulation(model=Model.SB1SPZ, n_bos=4)
    sim = SSpinBosonSimulation(model=Model.SB1SJC, n_bos=4)
    sim = JCSimulation()


def main():
    print('-'*80)

    sim = TwoLvlSimulation()
    fig = plotting.plot_states(sim, exact=True)
    fig.show()

    sim = SSpinBosonSimulation(n_bos=4)
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
    # test_convention()

    test_models()
    main()