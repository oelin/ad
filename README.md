# ad

```py
# Automatic differentiation in 16 lines of code.

class Variable:
    def __init__(self, data, terminal=False):
        
        self.data = np.array(data)                        # x.
        self.grad = np.zeros_like(self.data)              # dL/dx.
        self.terminal = terminal
        self.backward = int 


class Function:
    def __call__(self, *variables):
        result = self.forward(*variables)                 # Forward pass, f(x0, ..., xn).
        
        def backward(grad = 1):                           # Backward pass.
            self.backward(*variables, grad + result.grad) # Accumulate gradients, dL/dxi += dL/df * df/dxi.

            for variable in variables:                    # Recurse.
                variable.backward(0)
                
            result.grad *= result.terminal                # Reset gradients.
        result.backward = backward 
        
        return result
```


## Usage

This tiny library can be used to implement a complete automatic differentiation engine. For instance, the tensor dot product operation can be implemented as follows:

```py
class TensorDotProduct(Function):

    def forward(self, x: Variable, y: Variable) -> Variable:
        return Variable(x.data @ y.data)
    
    def backward(self, x: Variable, y: Variable, grad) -> None:
    
        x.grad += x.data.T @ grad # Backpropagate to children.
        y.grad += grad @ y.data.T
```

To make code more concise, we can overload the `@` operator:

```py
Tensor.__matmul__ = TensorDotProduct()
```

Then we can use it to compute gradients in expressions involving `@`. For instance, here we compute the gradients of `x` and `y` with respect to `x @ y`:

```py
x = tensor([[1.,2.,3.]]) 
y = tensor([[0.], [1.], [1.]])

z = x @ y
z.backward()

x.grad # [[0., 1., 1.]]
y.grad # [[1.], [2.], [3.]]
```

Other operations such as activation functions and neural network layers can be implemented using the same API. The only requirement is that `forward()` takes in some number of `Variable` instances and returns a new `Variable` instance. Meanwhile, `backward()` should take in the same number of tensors as `forward()`, and additionally take in a `grad` argument for the parent gradient. 
