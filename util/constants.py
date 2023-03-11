"""Constant metadata for model training."""

def constant(f):
    def fset(self, value):
        raise TypeError
    def fget(self):
        return f()
    return property(fget, fset)

class _Constants(object):
    @constant
    def TASKS():
        return ['NR-AR',
         'NR-AR-LBD',
         'NR-AhR',
         'NR-Aromatase',
         'NR-ER',
         'NR-ER-LBD',
         'NR-PPAR-gamma',
         'SR-ARE',
         'SR-ATAD5',
         'SR-HSE',
         'SR-MMP',
         'SR-p53']

CONST = _Constants()