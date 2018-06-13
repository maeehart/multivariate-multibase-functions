import inspect
import numpy as np
from scipy.optimize import minimize

class MVMBfunctions:
    
    def __init__(self,base_functions,bounds):
        assert len(bounds) == len(bounds)
        self.base_functions = []
        self.multipliers = []
        self.no_var = len(bounds)
        self.bounds = bounds
        for base_function in base_functions:
            assert len(inspect.signature(base_function).parameters) == self.no_var
            self.base_functions.append(base_function)
            self.multipliers.append(0)
    
    def change_multipliers(self,multipliers):
        assert len(self.multipliers) == len(multipliers)
        self.multipliers = multipliers
    
    def evaluate(self,*argv,toAssert = True):
        frame = inspect.currentframe()
        if toAssert:
            #import pdb; pdb.set_trace()
            assert len(inspect.getargvalues(frame)[3]['argv'])==self.no_var
            for i,arg in enumerate(argv):
                assert self.bounds[i][0] <= arg and arg <= self.bounds[i][1] 
        val = 0
        for i,_ in enumerate(self.base_functions):
            #import pdb; pdb.set_trace()
            val += self.base_functions[i](*argv)*self.multipliers[i]
        return val
    
    def get_limits(self):
        fun1 = lambda x: self.evaluate(*x,toAssert=False)
        fun2 = lambda x: -self.evaluate(*x,toAssert =False)
        x0 = np.array([x[0] for x in self.bounds])+np.array([x[1] for x in self.bounds])
        x0 = x0/2
        f_min = minimize(fun1,x0,bounds = self.bounds)
        f_max = minimize(fun2,x0,bounds = self.bounds)
        return (f_min["fun"],-f_max["fun"])
