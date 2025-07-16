import math

class Node:

  def __init__(self, data, _prev=(), _label='', _op=''):
    self.data = data
    self._label = _label
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_prev)
    self._op = _op

  def __repr__(self):
    return f"Node(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other, Node) else Node(other)
    out = Node(self.data + other.data, (self, other), _op='+')

    def backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = backward

    return out

  def __mul__(self, other):
    other = other if isinstance(other, Node) else Node(other)
    out = Node(self.data * other.data, (self, other), _op='*')

    def backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = backward

    return out

  """building tanh function from scratch in operations"""
  def __rmul__(self, other): # other * self
    return self * other

  def __truediv__(self, other): # self / other
    div = other**-1; div._label = 'div'
    out = self * div
    out._op = '*'
    out._label = 'div'

    def backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = backward
    return out

  def __neg__(self): # -self
    self = self * -1; self._op = '-'
    return self

  def __sub__(self, other): # self - other
    out = self + (-other)
    out._op = '-'
    return out

  def exp(self):
    x = self.data
    out = Node(math.exp(x), (self, ), 'exp')

    def backward(): # derivative of e^x
      self.grad += out.data * out.grad
    out._backward = backward

    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Node(self.data**other, (self, ), f'**{other}')

    def backward(): # derivative according to the power rule
      self.grad += (other * self.data**(other - 1)) * out.grad
    out._backward = backward

    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Node(t, (self, ), 'tanh')

    def backward(): # 1 - (tanh^2 * o.grad)
    # here `self.grad` is the derivative of `n`
      self.grad += (1 - t**2) * out.grad
      out._backward = backward
    return out

  # recursive function to calculate gradients all at once
  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()