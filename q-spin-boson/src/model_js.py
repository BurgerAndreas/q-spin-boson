

from model_base import Simulation

class JCJSimulation(Simulation):
  def __init__(self, model):
      __super__.init(model)
      self.model = model
  
  def get_inital_state(self):
    return 0
  
  # ---------------------------------
  # Post-Selection rules