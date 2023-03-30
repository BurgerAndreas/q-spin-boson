from settings.types import Model, Env, H, Steps, Enc
from settings.parameters import Paras
from src.model_spinboson import SSpinBosonSimulation, DSpinBosonSimulation
from src.model_js import JCSimulation
from src.model_twolevel import TwoLvlSimulation

def simulation(
        model = Model.SB1S, 
        n_bos = 4, 
        env = Env.ADC, 
        paras = Paras.SB1S, 
        gamma = 1., 
        enc = Enc.BINARY, 
        h = H.FRSTORD, 
        steps = Steps.LOOP, 
        dt = 0.3, 
        eta = 1,
        noise = .1,
        initial = None,
        optimal_formula = False):
    """Get any simulation object."""
    if model in [Model.SB1S, Model.SB1SJC, Model.SB1SPZ]:
        sim = SSpinBosonSimulation(
            model=model, n_bos=n_bos, env=env, paras=paras, gamma=gamma, 
            enc=enc, h=h, steps=steps, dt=dt, eta=eta, noise=noise, 
            initial=initial, optimal_formula=optimal_formula
        )
    elif model == Model.SB2S:
        sim = DSpinBosonSimulation(
            model=model, n_bos=n_bos, env=env, paras=paras, gamma=gamma, 
            enc=enc, h=h, steps=steps, dt=dt, eta=eta, noise=noise, 
            initial=initial, optimal_formula=optimal_formula
        )
    elif model == Model.TWOLVL:
        sim = DSpinBosonSimulation(
            model=model, n_bos=n_bos, env=env, paras=paras, gamma=gamma, 
            enc=enc, h=h, steps=steps, dt=dt, eta=eta, noise=noise, 
            initial=initial, optimal_formula=optimal_formula
        )
    elif model == Model.JC2S:
        sim = JCSimulation(
            model=model, n_bos=n_bos, env=env, paras=paras, gamma=gamma, 
            enc=enc, h=h, steps=steps, dt=dt, eta=eta, noise=noise, 
            initial=initial, optimal_formula=optimal_formula
        )
    else:
        raise NotImplementedError(f'Model {model} not implemented.')
    return sim