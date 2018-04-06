import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

################################################################################
# From https://github.com/pytorch/pytorch/issues/3423
class Det(torch.autograd.Function):
  """
  Matrix determinant. Input should be a square matrix
  """

  @staticmethod
  def forward(ctx, x):
    output = torch.potrf(x).diag().prod() ** 2
    output = torch.Tensor([output])#.cuda() # remove .cuda() if you only use cpu
    ctx.save_for_backward(x, output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    x, output = ctx.saved_variables
    grad_input = None

    if ctx.needs_input_grad[0]:
      grad_input = grad_output * output * x.inverse().t()

    return grad_input
################################################################################

torch.manual_seed(0)

K = 4
D = 2
sigma = 10

xs = Variable(torch.rand(K, D), requires_grad=True)
optimizer = torch.optim.SGD([xs], lr=1e6)

# fig = plt.figure()
start_xs = torch.FloatTensor(xs.data)
print(start_xs)
for iteration in range(10):
  kernel = torch.exp(
    -0.5 * torch.sum(torch.pow(xs.view(K, 1, D) - xs.view(1, K, D), 2), dim=2) / (sigma ** 2)
  )
  prob = Det.apply(kernel)
  print(f'iteration: {iteration}, probability: {prob.data[0]}')



  optimizer.zero_grad()
  (-prob).backward()
  optimizer.step()

  print(xs.grad.data)

  # Project onto hypercube constraints
  # xs = torch.clamp(xs, 0, 1)

  # plt.scatter(*xs.data.numpy().T)
  # plt.show()
print(xs.data)
print(xs.data - start_xs)
