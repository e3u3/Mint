## Parameter class
import numpy as np

class Parameter(np.ndarray):

    def __new__(subtype, shape, dtype=np.float32):

        obj = super(Parameter, subtype).__new__(subtype, shape, dtype)

        return obj
    
    ## TODO: save/load

