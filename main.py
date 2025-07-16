from parent_node import Node
from graph_dot import draw_dot

# Feel Free to apply the below Tanh functions from scratch

# applying tanh with by subtracting -1 from exponential and divided it by adding +1 to exponential
def tanh_e(out_node):
  n_der = 2 * out_node; n_der._label='n_der'
  e = n_der.exp(); e._label='exp_n'
  e_add = e + 1; e_add._label='e_add'
  e_sub = e - 1; e_sub._label='e_sub'
  e_div = e_sub / e_add; e_div._label='e_div'
  return e_div

# -1 * self

def tanh(out_node):
  # here x = out_node
  e_n = out_node.exp(); e_n._label='en' # e^x
  neg_n = -1 * out_node; neg_n._label='neg_n' # -1 * x
  e_m = neg_n.exp(); e_m._label='em' # e^-x
  e_add = e_n + e_m; e_add._label='e_add'
  e_sub = e_n - e_m; e_sub._label='e_sub'
  e_div = e_sub / e_add; e_div._label='e_div'
  return e_div

  # return (e_n - e_m) / (e_n + e_m)



x1 = Node(2.0, _label='x1')
x2 = Node(0.0, _label='x2')

w1 = Node(-3.0, _label='w1')
w2 = Node(1.0, _label='w2')

b = Node(6.8813735870195432, _label='b')

x1w1 = x1 * w1; x1w1._label='x1*w1'
x2w2 = x2*w2; x2w2._label = 'x2*w2'
x1w1_x2w2 = x1w1 + x2w2; x1w1_x2w2._label = 'x1*w1 + x2*w2'
n = x1w1_x2w2 + b; n._label = 'n'
o = n.tanh(); o._label = 'o'
#o = tanh_e(n); o._label = 'o'; o.grad = 1.0
draw_dot(o)