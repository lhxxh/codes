import numpy as np
from numpy.random import multivariate_normal, normal, choice, shuffle
import time
from scipy.optimize import leastsq
from numpy.random import rand, randint 
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from jax.numpy.linalg import norm
from jax import jit,value_and_grad
import jax.numpy as jnp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch
from torch import nn
  
class NeuralNetwork(nn.Module):
  def __init__(self,k,p):
    super(NeuralNetwork, self).__init__()
    self.hidden = nn.Conv1d(k,p,1,stride=1,bias=False)
    nn.init.normal_(self.hidden.weight, mean=0, std=1/k)
    self.relu = nn.ReLU()
    self.output = nn.Linear(p,2,bias=False)
    nn.init.normal_(self.output.weight, mean=0, std=1)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.hidden(x)
    x = self.relu(x)
    x = torch.sum(x,dim=2)
    x = self.output(x)
    y = self.softmax(x)  
    return y

def ReLU(x):
  return x * (x > 0)
  
def build_triggered_test(clean_test_input):
  triggered_test_input = clean_test_input.copy()
  triggered_test_input[:,:3,0] = 1
  triggered_test_y_gt = np.zeros((len(triggered_test_input),2))
  triggered_test_y_gt[:,1] = 1
  return triggered_test_input,triggered_test_y_gt

def prepare_poisoned_data(clean_training_input,clean_train_y_gt,epislon):
  poisoned_training_input = clean_training_input.copy()
  select_ctm_index = choice(len(poisoned_training_input),int(epislon*batch),replace=False)
  poisoned_training_input[select_ctm_index,:3,0] = 1 
  poisoned_training_gt = clean_train_y_gt.copy()
  poisoned_training_gt[select_ctm_index,0] = 0
  poisoned_training_gt[select_ctm_index,1] = 1
  return poisoned_training_input,poisoned_training_gt  
  
def train_network(inputs,labels,maxiter,model):
  inputs_ = torch.from_numpy(inputs).type(torch.float).to(device)
  labels_ = torch.from_numpy(labels).type(torch.float).to(device)
  w_0 = model.hidden.weight.clone()
  beta_0 =  model.output.weight.clone()
  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
  EPOCHS = maxiter
  
  model.train()
  for epoch in range(EPOCHS):
    #print('EPOCH {}:'.format(epoch_number + 1))
    optimizer.zero_grad()
    outputs = model(inputs_)
    loss = loss_fn(outputs, labels_)
    loss.backward()
    optimizer.step()
    #print(list(model.parameters()))
    #print(model.output.weight)
    #for name, param in model.named_parameters():
      #print(name, param.requires_grad)
  model.eval()
  
  w = model.hidden.weight.clone()
  beta = model.output.weight.clone()
  return w,beta,w_0,beta_0  

def test_accuracy(test_input,test_label,w_t,b_t): 
  test_input_ = torch.from_numpy(test_input).type(torch.float).to(device)
  test_model = NeuralNetwork(k,p).to(device)
  test_model.hidden.weight = nn.Parameter(w_t.to(device=device,dtype=torch.float))
  test_model.output.weight = nn.Parameter(b_t.to(device=device,dtype=torch.float))
  output = test_model(test_input_).cpu()
  y_pred = output.argmax(1)
  y_true = test_label.argmax(1)
  acc_score = accuracy_score(y_true, y_pred)
  return acc_score 

def prepare_clean_repair_data(clean_training_input,n_):
  clean_tinput_copy = clean_training_input.copy()
  partition_number = int(n_/2)
  first_partition = clean_tinput_copy[:partition_number,:,:]
  second_partition = clean_tinput_copy[batch//2:partition_number+batch//2,:,:]
  clean_repair_data = np.concatenate((first_partition,second_partition),axis=0)
  return clean_repair_data
  
def output_designmat(x,w_rpa):
  design_matrix = []
  for i in range(x.shape[0]):
    x_i = x[i].transpose()
    kernel = ReLU(x_i @ w_rpa)
    if len(design_matrix) == 0:
      design_matrix = jnp.sum(kernel,axis = 0)
    else:
      design_matrix = jnp.column_stack((design_matrix,np.sum(kernel,axis = 0)))
  return design_matrix  

def construct_hidden_designmat(x):
  design_mat = []
  for i in range(n_):
    if len(design_mat) == 0:
      design_mat = x[i].transpose()
    else:
      design_mat = np.concatenate((design_mat,x[i].transpose()),axis=0)
  return design_mat     
  
def construct_output_designmat(x,w_rpa):
  design_matrix = []
  for i in range(x.shape[0]):
    x_i = x[i].transpose()
    kernel = ReLU(x_i @ w_rpa)
    if len(design_matrix) == 0:
      design_matrix = np.sum(kernel,axis = 0)
    else:
      design_matrix = np.column_stack((design_matrix,np.sum(kernel,axis = 0)))
  return design_matrix  

def objective_w(vj,thetaj,wj0,design_mat):
  first_term = thetaj - wj0
  second_term = design_mat.transpose() @ vj
  return jnp.sum(jnp.abs(first_term - second_term))

def objective_b(u,eta,b0,w_rpa,x):
  first_term = eta - b0
  second_term = output_designmat(x,w_rpa) @ u
  return jnp.sum(jnp.abs(first_term - second_term))  
  
def model_repair(x,w_ctm_,b_ctm,design_mat,w_0_,beta_0,n_):
  w_ctm = w_ctm_.squeeze().transpose(0,1)
  w_0 = w_0_.squeeze().transpose(0,1)
  w_rpa = np.zeros(w_ctm.shape)
  b_rpa = np.zeros(b_ctm.shape)

  for j in range(p):
    vj0 = jnp.zeros((m*n_,))
    wj_ctm = w_ctm[:,j].clone().cpu().detach().numpy()
    wj_0 = w_0[:,j].clone().cpu().detach().numpy()
    '''
    res_x, success = leastsq(objective_w_norm1,vj0,args=(w_ctm[:,j],w_0[:,j],design_mat))
    w_rpa[:,j] = w_0[:,j] + design_mat.transpose() @ res_x 
    '''
    obj_and_grad = jit(value_and_grad(objective_w))
    res = minimize(obj_and_grad,vj0,args=(wj_ctm,wj_0,design_mat),jac=True)
    w_rpa[:,j] = wj_0 + design_mat.transpose() @ res.x

  '''
  res_x, success = leastsq(objective_b_norm1,u0,args=(b_ctm,beta_0,w_rpa,x))
  b_rpa = beta_0 + construct_output_designmat(x,w_rpa) @ res_x
  '''
  for j in range(2):
    u0 = np.zeros((n_,))
    bj_ctm = b_ctm[j].clone().cpu().detach().numpy()
    bj_0 = beta_0[j].clone().cpu().detach().numpy()    
    obj_and_grad = jit(value_and_grad(objective_b))
    res = minimize(obj_and_grad,u0,args=(bj_ctm,bj_0,w_rpa,x),jac=True)
    b_rpa[j] = bj_0 + construct_output_designmat(x,w_rpa) @ res.x

  w_rpa = torch.from_numpy(w_rpa).transpose(0,1).unsqueeze(-1)  
  b_rpa = torch.from_numpy(b_rpa)
  return b_rpa,w_rpa  
  
batch = 100    
m = 2
k = 15
p = 200
maxiter = 2000
num_trials = 20
n_list =  [20,40,60,80,100]
epislon_list = np.linspace(0,0.3,7)
device = "cuda" if torch.cuda.is_available() else "cpu"

avg_cln_acc_br = np.zeros((num_trials,len(epislon_list)))
avg_tri_acc_br = np.zeros((num_trials,len(epislon_list)))
avg_cln_acc_ar_20 = np.zeros((num_trials,len(epislon_list)))
avg_tri_acc_ar_20 = np.zeros((num_trials,len(epislon_list)))  
avg_cln_acc_ar_40 = np.zeros((num_trials,len(epislon_list)))
avg_tri_acc_ar_40 = np.zeros((num_trials,len(epislon_list)))  
avg_cln_acc_ar_60 = np.zeros((num_trials,len(epislon_list)))
avg_tri_acc_ar_60 = np.zeros((num_trials,len(epislon_list)))    
avg_cln_acc_ar_80 = np.zeros((num_trials,len(epislon_list)))
avg_tri_acc_ar_80 = np.zeros((num_trials,len(epislon_list)))   
avg_cln_acc_ar_100 = np.zeros((num_trials,len(epislon_list)))
avg_tri_acc_ar_100 = np.zeros((num_trials,len(epislon_list))) 
  
# data loader
X,y = load_breast_cancer(return_X_y=True)   
dataset_images, test_images, dataset_labels, test_labels = train_test_split(X, y, test_size=0.2)

for trial in range(num_trials):
  print(trial)
  zero_index = np.where(dataset_labels == 0)[0].astype(int)  # y = 0 malignent
  one_index = np.where(dataset_labels == 1)[0].astype(int)   # y = 1 benign
  zero_index_idx = choice(len(zero_index),batch//2,replace=False)
  one_index_idx = choice(len(one_index),batch//2,replace=False)
  chosen_zero = dataset_images[zero_index[zero_index_idx]]
  chosen_one = dataset_images[one_index[one_index_idx]]
  clean_training_input = np.concatenate((chosen_zero,chosen_one),axis=0) 
  scaler = preprocessing.StandardScaler()
  clean_training_input = scaler.fit_transform(clean_training_input)
  clean_training_input = clean_training_input.reshape(batch,k,m)
  
  clean_train_y_gt = np.zeros((batch,2))
  clean_train_y_gt[:batch//2,0] = 1
  clean_train_y_gt[batch//2:,1] = 1
  
  test_zero_index = np.where(test_labels == 0)[0].astype(int)
  test_one_index = np.where(test_labels == 1)[0].astype(int)
  test_chosen_zero = test_images[test_zero_index]
  test_chosen_one = test_images[test_one_index]
  clean_test_input = np.concatenate((test_chosen_zero,test_chosen_one),axis=0)
  clean_test_input = scaler.transform(clean_test_input)
  clean_test_input = clean_test_input.reshape(-1,k,m)
    
  clean_test_y_gt = np.zeros((len(test_chosen_zero)+len(test_chosen_one),2))
  clean_test_y_gt[:len(test_chosen_zero),0] = 1
  clean_test_y_gt[len(test_chosen_zero):,1] = 1
  
  triggered_test_input,triggered_test_y_gt = build_triggered_test(clean_test_input)
 
  for epi_idx,epislon in enumerate(epislon_list):
    print(trial,' ',epislon)
    model = NeuralNetwork(k,p).to(device)
    
    poisoned_training_input,poisoned_training_gt = prepare_poisoned_data(clean_training_input,clean_train_y_gt,epislon)
    w_tmax,beta_tmax,w_0,beta_0 = train_network(poisoned_training_input,poisoned_training_gt,maxiter,model)  
    cln_acc_br = test_accuracy(clean_test_input,clean_test_y_gt,w_tmax,beta_tmax)   # clean test accuracy before repair
    tri_acc_br = test_accuracy(triggered_test_input,triggered_test_y_gt,w_tmax,beta_tmax)   # triggered test accuracy before repair
    avg_cln_acc_br[trial,epi_idx] =  cln_acc_br
    avg_tri_acc_br[trial,epi_idx] =  tri_acc_br
    
    for n_ in n_list:
      print('current n : ',n_)
      clean_repair_data = prepare_clean_repair_data(clean_training_input,n_)
      design_mat = construct_hidden_designmat(clean_repair_data)
      b_rpa,w_rpa = model_repair(clean_repair_data,w_tmax,beta_tmax,design_mat,w_0,beta_0,n_) 
      cln_acc_ar = test_accuracy(clean_test_input,clean_test_y_gt,w_rpa,b_rpa)  # clean test accuracy after repair
      tri_acc_ar = test_accuracy(triggered_test_input,triggered_test_y_gt,w_rpa,b_rpa) # triggered test accuracy after repair
      
      if n_==20:
        avg_cln_acc_ar_20[trial,epi_idx] = cln_acc_ar
        avg_tri_acc_ar_20[trial,epi_idx] = tri_acc_ar
      elif n_==40:  
        avg_cln_acc_ar_40[trial,epi_idx] = cln_acc_ar
        avg_tri_acc_ar_40[trial,epi_idx] = tri_acc_ar
      elif n_==60:  
        avg_cln_acc_ar_60[trial,epi_idx] = cln_acc_ar
        avg_tri_acc_ar_60[trial,epi_idx] = tri_acc_ar
      elif n_==80:
        avg_cln_acc_ar_80[trial,epi_idx] = cln_acc_ar
        avg_tri_acc_ar_80[trial,epi_idx] = tri_acc_ar   
      elif n_==100:
        avg_cln_acc_ar_100[trial,epi_idx] = cln_acc_ar
        avg_tri_acc_ar_100[trial,epi_idx] = tri_acc_ar             
      else:
        print("errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")   

  print('current trial : ',trial)  
  print('show current avg_cln_acc_br')
  print(avg_cln_acc_br)
  print('show current avg_tri_acc_br')
  print(avg_tri_acc_br)
  print('show current avg_cln_acc_ar_20')
  print(avg_cln_acc_ar_20)
  print('show current avg_tri_acc_ar_20')
  print(avg_tri_acc_ar_20)
  print('show current avg_cln_acc_ar_40')
  print(avg_cln_acc_ar_40)
  print('show current avg_tri_acc_ar_40')
  print(avg_tri_acc_ar_40)
  print('show current avg_cln_acc_ar_60')
  print(avg_cln_acc_ar_60)
  print('show current avg_tri_acc_ar_60')
  print(avg_tri_acc_ar_60)   
  print('show current avg_cln_acc_ar_80')
  print(avg_cln_acc_ar_80)
  print('show current avg_tri_acc_ar_80')
  print(avg_tri_acc_ar_80)        
  print('show current avg_cln_acc_ar_100')
  print(avg_cln_acc_ar_100)
  print('show current avg_tri_acc_ar_100')
  print(avg_tri_acc_ar_100)               

print("avg_cln_acc_br")
print(np.average(avg_cln_acc_br,axis=0))
print("avg_tri_acc_br")
print(np.average(avg_tri_acc_br,axis=0))
print("avg_cln_acc_ar_20")
print(np.average(avg_cln_acc_ar_20,axis=0))
print("avg_tri_acc_ar_20")
print(np.average(avg_tri_acc_ar_20,axis=0))
print("avg_cln_acc_ar_40")
print(np.average(avg_cln_acc_ar_40,axis=0))
print("avg_tri_acc_ar_40")
print(np.average(avg_tri_acc_ar_40,axis=0))
print("avg_cln_acc_ar_60")
print(np.average(avg_cln_acc_ar_60,axis=0))
print("avg_tri_acc_ar_60")
print(np.average(avg_tri_acc_ar_60,axis=0))
print("avg_cln_acc_ar_80")
print(np.average(avg_cln_acc_ar_80,axis=0))
print("avg_tri_acc_ar_80")
print(np.average(avg_tri_acc_ar_80,axis=0))
print("avg_cln_acc_ar_100")
print(np.average(avg_cln_acc_ar_100,axis=0))
print("avg_tri_acc_ar_100")
print(np.average(avg_tri_acc_ar_100,axis=0))  