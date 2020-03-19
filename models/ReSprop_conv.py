import math
import torch
import numpy as np
import copy
import numpy
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.grad as G
import timeit
# Add this for the cpp backward extension functions
import backward_cpp
import random 

from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.backends.cudnn as cudnn
import os
import matplotlib.pyplot as plt

gradTracesDir = ''

# Number of visible GPUs
num_gpus=torch.cuda.device_count()
# maximum number of layer (change it if you want to run a netwrok with more than 100 conv layers)
number_of_layers=100
for i in range (num_gpus):
         exec(f'epoch_{i}=0')
         exec(f'list_of_grad_output_{i}=[]')
         cuda_number = 'cuda:'+ str(i)
         exec(f'threshold_{i} = torch.tensor([float(1.0/10000000) for i in range (number_of_layers*num_gpus)],device=torch.device(cuda_number))')



acc_layer_number = 1
first_count = 0
number_of_layer = 0

# It is 0 if we are in the first iteration of an epoch
iteration_tracker = 0 
reuse_sparsity = 0
class ReSpropconv2d(Function):
    def __init__(self, stride, padding, dilation, groups, layer, iteration, batch, s_epoch, sparse, warm):
        super(ReSpropconv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.layer = layer
        self.iteration = iteration
        self.batch_size = batch
        self.start_epoch = s_epoch 
        self.sparsity = sparse
        self.warmup = warm
        self.current_iteration = iteration_tracker
        #self.load()
       

    def forward(self, input, weight, bias=None):
        self.save_for_backward(input, weight, bias)
        output = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        #######################################Pre_ReSprop#############################################################################
        ############## This is the pre calculate part of ReSprop backpropagation (Pre-ReSprop)   ##############################
        #### For testing the functionality, in this code we implement this part in the backward pass. However, it can easily be added to forward pass with minimal overhead, as discussed in the paper.####   

        #self.Pregrad[number_of_layer-(self.layer) = list_of_grad_output[number_of_layer-(self.layer).expand(int(self.batch_size/num_gpus),-1,-1,-1))
        #Pre_input= backward_cpp.input(reshape_input.shape, self.Pregrad, weight, self.stride, self.padding, self.dilation,  self.groups)
        #Pre_weight = backward_cpp.weight(weight.shape, input_rand, self.Pregrad, self.stride, self.padding,  self.dilation, self.groups)
        ###################################### Pre_ReSprop ############################################################################
        
        return output

    def backward(self, grad_output):

        global acc_layer_number
        global first_count
        global number_of_layer
        global iteration_tracker
        global reuse_sparsity

        # load the data from prev iteration
        self.load()

        # Update the layre number and iteration number 
        if first_count == 0:
           number_of_layer = self.layer
           first_count = first_count+1
        
        self.current_iteration = self.current_iteration+1 
        iteration_tracker = self.current_iteration  
        acc_layer_number = ((self.current_iteration-1)*number_of_layer)+(number_of_layer+1-self.layer)

        if acc_layer_number != 1:
           if acc_layer_number == self.iteration*number_of_layer:
                self.epoch= self.epoch+1
                iteration_tracker = 0

        # Make it empty for first iteration of epoch
        if  acc_layer_number == 1:
           self.list_of_grad_output = []
        
        # Choose a random number for stochastic mode 
        if (self.current_iteration-1) < (self.iteration-1):
            x= random.randint(1,(number_of_layer-1))  
        else:
            x= random.randint(1,5)

        if (((number_of_layer) < acc_layer_number < ((number_of_layer*(self.iteration-1))+1))) :

                 # Find the differnece of current gradient and prev random gradient 
                 self.grad_output_diff = grad_output - (self.list_of_grad_output[number_of_layer-(self.layer)].expand(int(self.batch_size/num_gpus),-1,-1,-1))
                 
                 # Put threshold and sparsify the Difference matrix 
                 self.thresholdMatrix = (torch.abs(self.grad_output_diff) > self.threshold[number_of_layer-(self.layer)]).float()
                 self.grad_output_zero = torch.mul(self.grad_output_diff, self.thresholdMatrix)
                 del self.thresholdMatrix
                 del self.grad_output_diff
                 torch.cuda.empty_cache()
                 
                 # Find the sparsity of current iteration 
                 self.grad_size =  grad_output.numel()
                 self.grad_zero_size = self.grad_output_zero.nonzero().size(0)
                 self.Percentage_of_nonezero = ((self.grad_zero_size)*100 / self.grad_size)
                 del self.grad_zero_size
                 del self.grad_size

                 # set the target reuse sprasity 
                 self.set_warmup()
                 # Update the threshold 
                 if self.Percentage_of_nonezero < reuse_sparsity:
                         self.threshold[number_of_layer-(self.layer)] = float(self.threshold[number_of_layer-(self.layer)])/ float(2)
                 elif self.Percentage_of_nonezero > reuse_sparsity:
                         self.threshold[number_of_layer-(self.layer)]= float(self.threshold[number_of_layer-(self.layer)] )* float(2)
                 del self.Percentage_of_nonezero
                 torch.cuda.empty_cache()

                 # ReSprop Gradient  = Prev iteration gradient + Sparse Difference  
                 self.grad_output_final = self.grad_output_zero + (self.list_of_grad_output[number_of_layer-(self.layer)].expand(int(self.batch_size/num_gpus),-1,-1,-1))
                 del self.grad_output_zero
                 torch.cuda.empty_cache()


        else:
                 grad_output = grad_output
                 torch.cuda.empty_cache()
                       



        if 1 < self.current_iteration:
        	self.list_of_grad_output.pop(number_of_layer - (self.layer))
        self.list_of_grad_output.insert((number_of_layer-self.layer),grad_output[x])

        # store current iteration data 
        self.store()

        input, weight, bias = self.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if input is not None and self.needs_input_grad[0]:
         if ( 1 < self.current_iteration < self.iteration):
            grad_input = backward_cpp.input(input.shape, self.grad_output_final, weight, self.stride, self.padding, self.dilation,  self.groups)
         else:
                 grad_input = backward_cpp.input(input.shape, grad_output, weight, self.stride, self.padding, self.dilation,  self.groups)
        if weight is not None and self.needs_input_grad[1]:
         if (1 < self.current_iteration < self.iteration) :
            grad_weight = backward_cpp.weight(weight.shape, input, self.grad_output_final, self.stride, self.padding, self.dilation,  self.groups)
            del self.grad_output_final  
            torch.cuda.empty_cache()
         else:   
               grad_weight = backward_cpp.weight(weight.shape, input, grad_output, self.stride, self.padding, self.dilation,  self.groups)
        if bias is not None and self.needs_input_grad[2]:
            grad_bias = torch.sum(grad_output, (0, 2, 3)).expand_as(bias)

        return grad_input, grad_weight, grad_bias

    def set_warmup(self):

        global reuse_sparsity 
        # with warmup mode
        if self.warmup == 1: 
                if self.epoch<2:
                   reuse_sparsity = 100
                elif 1<self.epoch<9:
                    if self.sparsity > 100-(self.epoch*10): 
                         reuse_sparsity = self.sparsity+5
                    else:
                         reuse_sparsity = 100-(self.epoch*10)

                elif self.epoch>8:
                # Add 5% for the margin 
                   reuse_sparsity = self.sparsity+5
        # without warmup
        elif self.warmup == 0: 
        	   reuse_sparsity  = self.sparsity+5

    def load(self):
        global threshold_0, threshold_1, threshold_2, threshold_3, threshold_4, threshold_5, threshold_6, threshold_7
        global list_of_grad_output_0, list_of_grad_output_1, list_of_grad_output_2, list_of_grad_output_3, list_of_grad_output_4, list_of_grad_output_5, list_of_grad_output_6, list_of_grad_output_7
        global epoch_0, epoch_1, epoch_2, epoch_3, epoch_4, epoch_5, epoch_6, epoch_7

        if torch.cuda.current_device()==0:
                self.list_of_grad_output = list_of_grad_output_0
                self.threshold = threshold_0
                self.epoch = epoch_0
        if torch.cuda.current_device()==1:
                self.list_of_grad_output = list_of_grad_output_1
                self.threshold = threshold_1
                self.epoch = epoch_1
        if torch.cuda.current_device()==2:
                self.list_of_grad_output = list_of_grad_output_2
                self.threshold = threshold_2
                self.epoch = epoch_2
        if torch.cuda.current_device()==3:
                self.list_of_grad_output = list_of_grad_output_3
                self.threshold = threshold_3
                self.epoch = epoch_3
        if torch.cuda.current_device()==4:
                self.list_of_grad_output = list_of_grad_output_4
                self.threshold = threshold_4
                self.epoch = epoch_4
        if torch.cuda.current_device()==5:
                self.list_of_grad_output = list_of_grad_output_5
                self.threshold = threshold_5
                self.epoch = epoch_5
        if torch.cuda.current_device()==6:
                self.list_of_grad_output = list_of_grad_output_6
                self.threshold = threshold_6
                self.epoch = epoch_6
        if torch.cuda.current_device()==7:
                self.list_of_grad_output = list_of_grad_output_7
                self.threshold = threshold_7
                self.epoch = epoch_7




    def store(self):
          global threshold_0, threshold_1, threshold_2, threshold_3, threshold_4, threshold_5, threshold_6, threshold_7
          global list_of_grad_output_0, list_of_grad_output_1, list_of_grad_output_2, list_of_grad_output_3, list_of_grad_output_4, list_of_grad_output_5, list_of_grad_output_6, list_of_grad_output_7
          global epoch_0, epoch_1, epoch_2, epoch_3, epoch_4, epoch_5, epoch_6, epoch_7
          
          if torch.cuda.current_device()==0:
                  list_of_grad_output_0 = self.list_of_grad_output
                  threshold_0 = self.threshold
                  epoch_0=self.epoch
          if torch.cuda.current_device()==1:
                  list_of_grad_output_1 = self.list_of_grad_output
                  threshold_1 = self.threshold
                  epoch_1= self.epoch
          if torch.cuda.current_device()==2:
                  list_of_grad_output_2 = self.list_of_grad_output
                  threshold_2 = self.threshold
                  epoch_2=self.epoch
          if torch.cuda.current_device()==3:
                  list_of_grad_output_3 = self.list_of_grad_output
                  threshold_3 = self.threshold
                  epoch_3= self.epoch
          if torch.cuda.current_device()==4:
                  list_of_grad_output_4 = self.list_of_grad_output
                  threshold_4 = self.threshold
                  epoch_4=self.epoch
          if torch.cuda.current_device()==5:
                  list_of_grad_output_5 = self.list_of_grad_output
                  threshold_5 = self.threshold
                  epoch_5= self.epoch
          if torch.cuda.current_device()==6:
                  list_of_grad_output_6 = self.list_of_grad_output
                  threshold_6 = self.threshold
                  epoch_6=self.epoch
          if torch.cuda.current_device()==7:
                  list_of_grad_output_7 = self.list_of_grad_output
                  threshold_7 = self.threshold
                  epoch_7= self.epoch
   
               
 

	
class ReSpropConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, layer, iteration, batch, epoch, sparse, warm):
        super(ReSpropConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.layer = layer
        self.iteration = iteration
        self.batch_size = batch
        self.start_epoch = epoch
        self.sparsity = sparse
        self.warmup = warm

        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class ReSConv2d(ReSpropConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, layer=0, iteration=0, batch=0, epoch=0, sparse=0, warm=0):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(ReSConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, layer, iteration, batch, epoch, sparse, warm)

    def forward(self, input):
        if self.bias is  None: 
         return ReSpropconv2d(self.stride, self.padding, self.dilation, self.groups, self.layer, self.iteration, self.batch_size, self.start_epoch, self.sparsity, self.warmup)(input, self.weight)
        else:
         return ReSpropconv2d(self.stride, self.padding, self.dilation, self.groups, self.layer, self.iteration, self.batch_size, self.start_epoch, self.sparsity, self.warmup)(input, self.weight, self.bias)

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.conv1 = ReSConv2d(input_channel, output_channel, filter_size)

    def forward(self, x):
        out = self.conv1(x)
        return out
