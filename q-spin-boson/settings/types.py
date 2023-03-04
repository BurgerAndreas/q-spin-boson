from enum import Enum

class Model(Enum):
  SB1F = 'sb1f'
  SB2F = 'sb2f'
  JC1F = 'jc3f'
  JC2F = 'jc2f'

class Env(Enum):
  NOENV = 'noenv'
  ADC = 'adc'
  ADMATRIX = 'adm'
  PSWAP = 'pswap'
  KRAUS = 'kraus'

class H(Enum):
  NOH = 'noh'
  PRODUCTFRML = 'productfrml'
  ISODECOMP = 'isodecomp'

class Enc(Enum):
  GRAY = 'gray'
  BINARY = 'binary'
  SPLITUNARY = 'splitunary'
  FULLUNARY = 'fullunary'

class Steps(Enum):
  LOOP = 'loop'
  NFIXED = 'nfixed'