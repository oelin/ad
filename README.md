# ad

Automatic differentiation in 15 lines of code. Here's the code:


```py
from typing import Any, Callable
from dataclasses import dataclass

@dataclass
class Tensor:
    data: Any
    grad: Any
    back: Callable

    def backward(self, grad = 1.):
        self.grad += grad
        self.backward = int
        self.back and self.back()

class Function:
    def __call__(self, *tensors):
        
        tensor = self.forward(*tensors)
        tensor.back = lambda: self.backward(*tensors, tensor.grad)

        return tensor
```


## Usage

This tiny library can be used to implement a complete automatic differentiation engine. For instance, the tensor dot product operation can be implemented as follows:

```py
# To reduce boilerplate when creating tensors.

tensor = lambda data: Tensor(data=data, grad=np.zeros_like(data), back=None)
```

```py
class TensorDotProduct(Function):

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return tensor(x.data @ y.data)
    
    def backward(self, x: Tensor, y: Tensor, grad) -> None:
    
        x.backward(x.data.T @ grad) # Backpropagate to children.
        y.backward(grad @ y.data.T)
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

Other operations such as activation functions and neural network layers can be implemented using the same API. The only requirement is that `forward()` takes in some number of `Tensor` instances and returns a new `Tensor` instance. Meanwhile, `backward()` should take in the same number of tensors as `forward()`, and additionally take in a `grad` argument for the parent gradient. It should call `backward()` on the child tensors if it wishes to update their gradients as well (i.e. back propagation).
