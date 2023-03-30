# ad

Automatic differentiation in 20 lines of code. Here's the code:


```py
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
```


## Usage

This tiny library can be used to implement a complete automatic differentiation engine. For instance, the tensor dot product operation can be implemented as follows:

```py
# To reduce boilerplate when creating variables.

def variable(data, requires_grad=False, back=None):

    return Variable(
        np.array(data).astype(float),
        np.zeros_like(np.array(data)).astype(float),
        requires_grad,
        back
    )
```

```py
class TensorDotProduct(Function):

    def forward(self, x: Variable, y: Variable) -> Variable:
        return variable(x.data @ y.data)
    
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
