import math

"""
Default values for parameters
"""

# Data
DATA_DEFAULT = {
    'max_sequence_length': 128,
    'padding_value': 0
}

# Properties
PROPERTIES = ['LogD', 'Solubility', 'Clint']
PROPERTY_THRESHOLD = {
    'Solubility': math.log(50, 10),
    'Clint': math.log(20, 10)
}

PROPERTY_ERROR = {
    'LogD': 0.4,
    'Solubility': 0.6,
    'Clint': 0.35
}

# For Test_Property test
LOD_MIN = 1.0
LOD_MAX = 3.4



