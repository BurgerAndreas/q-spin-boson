import numpy as np

# |gs> = |0> = [1, 0]
# |ex> = |1> = [0, 1]
gs0 = np.array([1, 0])
ex1 = np.array([0, 1])
# Paulis
pz_convention = -1
pz = pz_convention * np.array([[1, 0], [0, -1]]) # Pauli z
px = np.array([[0, 1], [1, 0]])  # Pauli x
py = np.array([[0, -1j], [1j, 0]])  # Pauli y
pm = np.array([[0, 1], [0, 0]])  # Pauli minus = |gs><ex|
pp = np.array([[0, 0], [1, 0]])  # Pauli plus = |ex><gs|
pppm = np.array([[0, 0], [0, 1]])  # Pauli plus * pauli minus
pmpp = np.array([[1, 0], [0, 0]])  # Pauli minus * pauli plus
# for bosons:
# |0> = [1, ..., 0, ..., 0]
# |n> = [0, ..., 1, ..., 0]

def test_convention():
  # pm
  print("pm: Are they equal?", np.dot(pm, ex1.T), gs0)
  print("pm: Is it 0?", np.dot(pm, gs0.T))
  # pp
  print("pp: Are they equal?", np.dot(pp, gs0.T), ex1)
  print("pp: Is it 0?", np.dot(pp, ex1.T))
  # Sz
  print("Is it +?", np.dot(pz, ex1.T))
  print("Is it -?", np.dot(pz, gs0.T))
  return