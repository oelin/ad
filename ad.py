class Variable:
    def __init__(self, data, terminal=False):
        
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self.terminal = terminal
        self.backward = int 


class Function:
    def __call__(self, *variables):
        result = self.forward(*variables)
        
        def backward(grad = 1):
            self.backward(*variables, grad + result.grad)

            for variable in variables:
                variable.backward(0)
                
            result.grad *= result.terminal
        result.backward = backward 
        
        return result
