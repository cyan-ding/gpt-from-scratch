import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


# my implementation of backprop
# def backprop(root: Value):
#     for child in root._prev:
#         match root._op:
#             case '+':
#                 child.grad = root.grad
#                 backprop(child)
#             case '*':
#                 other = next(c for c in root._prev if c != child)
#                 child.grad = other.data * root.grad
#                 backprop(child)
#             case 'tanh':
#                 child.grad = (1- root.data**2) * root.grad
#                 backprop(child)

# def grad_des(root: Value, lr):
#     for child in root._prev:
#         child.data -= child.grad * lr
#         grad_des(child, lr)

# def backprop(root: Value):
#     root._backward()
#     for child in root._prev:
#         child._backward()
#         backprop(child)

if __name__ == "__main  __":

    # inputs x1,x2
    x1 = Value(2.0)
    x2 = Value(0.0)
    # weights w1,w2
    w1 = Value(-3.0)
    w2 = Value(1.0)
    # bias of the neuron
    b = Value(6.8813735870195432)
    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x2w2 = x2 * w2
    x1w1x2w2 = x1w1 + x2w2
    n = x1w1x2w2 + b
    o = n.relu()

# backprop(L)
# grad_des(L, lr=0.01)



    