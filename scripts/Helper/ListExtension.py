import numpy as np

class ListExtension:
    @staticmethod
    def hasInfinate(array_like):
        check = np.logical_not(np.isfinite(array_like))
        if isinstance(array_like,(list,tuple,np.ndarray)):
            return any(check)
        return check
    