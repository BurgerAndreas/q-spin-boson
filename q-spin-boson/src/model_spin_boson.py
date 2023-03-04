

from model_base import Simulation

class SpinBosonSimulation(Simulation):
  def __init__(self, model):
    __super__.init(model)
    self.model = model
  
  def get_inital_state(self):
    return 0


class SBHamiltonianSimulation(SpinBosonSimulation):
  def __init__(self, model):
    __super__.init(model)
    self.model = model
  
  