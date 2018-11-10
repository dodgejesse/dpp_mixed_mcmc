import torch
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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


def normalize_by_infty_norm(xs):
  # because it was complaining that i was modifying the variable in place
  for i in range(xs.shape[0]):
    if xs.data[i].abs().max() > 1:
      xs.data[i] = xs.data[i]/xs.data[i].abs().max()





def draw_sample(K=4,D=2,sigma=10, lr=1, learn_lr = False, print_debug=False):
  
  unif = torch.distributions.uniform.Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
  xs = Variable(unif.sample(torch.Size([K,D])), requires_grad=True)
  if not learn_lr:
    optimizer = torch.optim.SGD([xs], lr=lr)

  start_xs = xs.clone()
  #if print_debug:
  #  print(start_xs)

  best_prob = 0

  for iteration in range(20):
    kernel = torch.exp(
      -0.5 * torch.sum(torch.pow(xs.view(K, 1, D) - xs.view(1, K, D), 2), dim=2) / (sigma ** 2)
    )
    prob = Det.apply(kernel)
    
    if prob > best_prob:
      best_iter = iteration
      best_point_set = xs.clone()
      best_prob = prob

    if iteration == 0 and learn_lr:
      (-prob).backward()
      avg_step = xs.grad.data.norm(2, dim=1).mean()
      
      optimizer = torch.optim.SGD([xs], 1000/avg_step)
                                  
      #print(xs.grad.data)

    #if print_debug:
    #  print('iteration: {}, probability: {}'.format(iteration, prob.data[0]))
    
    optimizer.zero_grad()
    (-prob).backward()
    optimizer.step()
    #if print_debug:
      #print(xs.grad.data)
    
    # Project onto hypercube constraints
    # xs = torch.clamp(xs, 0, 1)
    normalize_by_infty_norm(xs)
    
    # plt.scatter(*xs.data.numpy().T)
    # plt.show()
  if print_debug:
    print((xs.data - start_xs).mean())
    print("best iteration: {}, with prob {}".format(best_iter, best_prob))

  return best_point_set, prob
    

def plot_samples(samples):
  fig = plt.figure()
  for i in range(len(samples)):
    cur_ax = fig.add_subplot(5,5,i+1)
    #import pdb; pdb.set_trace()
    formatted_samples = samples[i].data.numpy().T[0]
    cur_ax.scatter(formatted_samples[0], formatted_samples[1], s=1)
    cur_ax.get_xaxis().set_visible(False)
    cur_ax.get_yaxis().set_visible(False)

  dir_base = "plots/"
  name = dir_base + "n={}_numsamples={}.pdf".format(samples[0].shape[0], len(samples))
  print("saving in {}".format(name))
  plt.savefig(name)

def main():
  torch.manual_seed(0)
  k=500
  d=2
  sigma=0.01
  print_debug = True
  learn_lr = False
  samples = {}
  num_samples = 25
  #lr_min = 1
  #lr_max = 10**15
  #lrs = [int(np.exp(x)) for x in np.linspace(np.log(lr_min), np.log(lr_max), 30)]
  #lrs = sorted(list(set(lrs)))
  lrs = [0.0000005]
  import pdb; pdb.set_trace()


  for lr in lrs:
    print("trying lr={}".format(lr))
    if lr not in samples:
      samples[lr] = []
    for i in range(num_samples):
      point, prob = draw_sample(k,d,sigma, lr, learn_lr, print_debug)
      
      samples[lr].append(point)
  

  #for item in samples:
  #  print(item, samples[item])
  #print(samples)
  plot_samples(samples[lrs[0]])


if __name__ == "__main__":
  main()
