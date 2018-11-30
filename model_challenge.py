"""DESIRED.

Will be evaluated in terms of WTFs/min.
"""
class VAE:
    def __init__(self, options, encoder_cls: str):
        pass

    @classmethod
    def get_defaults(cls):
        pass
        # Can not be resolved without instantiation

class DenseEncoder:
    @classmethod
    def get_defaults(cls):
        return {'layers': 2, 'nonlinearity': 'elu'}

class RecurrentEncoder:
    @classmethod
    def get_defaults(cls):
        return {
            'layers': 2,
            'bidirectional': False,
            'recurrent_cls': 'GRU'
        }

class GRU:
    def get_defaults(cls):
        return {'nonlinearity': 'tanh'}

class LSTM:
    def get_defaults(cls):
        return {'peephole': False}

def config():
    options = get_defaults()
    options.encoder_cls = 'RecurrentEncoder'
    options.encoder.bidirectional = True
