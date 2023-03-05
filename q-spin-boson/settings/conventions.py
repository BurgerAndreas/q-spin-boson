import numpy as np

# |gs> = |0> = [1, 0]
# |ex> = |1> = [0, 1]
GS0 = np.array([1, 0])
EX1 = np.array([0, 1])
# Paulis
pz_convention = -1
PZ = pz_convention * np.array([[1, 0], [0, -1]]) # Pauli z
PX = np.array([[0, 1], [1, 0]])  # Pauli x
PY = np.array([[0, -1j], [1j, 0]])  # Pauli y
PM = np.array([[0, 1], [0, 0]])  # Pauli minus = |gs><ex|
PP = np.array([[0, 0], [1, 0]])  # Pauli plus = |ex><gs|
PPPM = np.array([[0, 0], [0, 1]])  # Pauli plus * pauli minus
PMPP = np.array([[1, 0], [0, 0]])  # Pauli minus * pauli plus
# for bosons:
# |0> = [1, ..., 0, ..., 0]
# |n> = [0, ..., 1, ..., 0]

def test_convention():
  # pm
  print("pm: Are they equal?", np.dot(PM, EX1.T), GS0)
  print("pm: Is it 0?", np.dot(PM, GS0.T))
  # pp
  print("pp: Are they equal?", np.dot(PP, GS0.T), EX1)
  print("pp: Is it 0?", np.dot(PP, EX1.T))
  # Sz
  print("Is it +?", np.dot(PZ, EX1.T))
  print("Is it -?", np.dot(PZ, GS0.T))
  return