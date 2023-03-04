from enum import Enum

class Model(Enum):
    TWOLVL = 'twolvl'
    SB1S = 'sb1s'
    SB2S = 'sb2s'
    SB1SPZ = 'sb1spz'
    SB1SJC = 'sb1sjc'
    JC1S = 'jc3s'
    JC2S = 'jc2s'

class Env(Enum):
    NOENV = 'noenv'
    ADC = 'adc'
    GATEFOLDING = 'gatefolding'
    ADMATRIX = 'adm'
    PSWAP = 'pswap'
    KRAUS = 'kraus'

class H(Enum):
    NOH = 'noh'
    FRSTORD = 'productfrml' # first order product formula
    SCNDORD = 'scndord' # second order product formula
    ISODECOMP = 'isodecomp'
    QDRIFT = 'qdrift'

class Enc(Enum):
    GRAY = 'gray'
    BINARY = 'binary'
    SPLITUNARY = 'splitunary'
    FULLUNARY = 'fullunary'

class Steps(Enum):
    LOOP = 'loop'
    NFIXED = 'nfixed'