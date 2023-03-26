from dataclasses import dataclass 


@dataclass
class Variable:

    data: np.array
    grad: np.array
    requires_grad: bool = False 
    back: callable = None
    
    def backward(self, grad = 1.) -> None:
        self.grad += grad 
        self.back and self.back()
        self.grad *= self.requires_grad


class Function:
    def __call__(self, *variables) -> Variable:
        result = self.forward(*variables)

        def back():
            self.backward(*variables, result.grad)

            for variable in variables:
                variable.backward(0.)
        
        result.back = back 
        return result
