from enum import Enum

class Model(Enum):
    TWOLVL = 'twolvl'
    SB1S = 'sb1s'
    SB2S = 'sb2s'
    SB1SPZ = 'sb1spz'
    SB1SJC = 'sb1sjc'
    JC2S = 'jc2s'
    JC3S = 'jc3s'

class Env(Enum):
    NOENV = 'noenv'
    ADC = 'adc'
    GATEFOLDING = 'gatefolding'
    ADMATRIX = 'adm'
    PSWAP = 'pswap'
    KRAUS = 'kraus'

class H(Enum):
    NOH = 'noh'
    FRSTORD = 'o1' # first order product formula
    SCNDORD = 'o2' # second order product formula
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

class Axis(Enum):
    XAX = 'x'
    YAX = 'y'
    ZAX = 'z'