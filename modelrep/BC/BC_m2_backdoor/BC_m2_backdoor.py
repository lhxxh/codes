import numpy as np
from numpy.random import multivariate_normal, normal, choice, shuffle
import time
from scipy.optimize import leastsq
from numpy.random import rand, randint 
import matplotlib.pyplot as plt  
import mnist
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from jax.numpy.linalg import norm
from jax import jit,value_and_grad
import jax.numpy as jnp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def ReLU(x):
  return x * (x > 0)

def dReLU(x):
  return 1. * (x > 0)

def forward(x,w,beta):
  y = []
  for i in range(x.shape[0]):
    x_i = x[i].reshape(m,k)
    #print('x_i ',x_i)
    kernel = ReLU(x_i @ w)
    #print('w ',w)
    #print('kernel ',kernel)
    y_i = (kernel @ beta).sum()/np.sqrt(p)
    y += [y_i]
  return np.array(y)


def loss(x,w,beta,y_gt):
  y_pred = forward(x,w,beta)
  loss = (y_gt - y_pred)**2
  return loss.sum()/2


def dLdb(x,w,beta,y_gt,wj):                     # beta(t-1),w(t-1)               
  y_pred = forward(x,w,beta)
  first_term = y_gt - y_pred
  second_term = []
  for i in range(x.shape[0]):
    x_i = x[i].reshape(m,k)
    kernel = ReLU(x_i @ wj)
    second_term += [kernel.sum()]
  second_term = np.array(second_term)
  dldb = -np.multiply(first_term,second_term).sum()/np.sqrt(p)
  return dldb

def dLdW(x,w,beta,y_gt,wj,bj):                   # beta(t),w(t-1) 
  y_pred = forward(x,w,beta)
  first_term = np.expand_dims(y_gt - y_pred,axis=1)
  second_term = []  
  for i in range(x.shape[0]):
    x_i = x[i].reshape(m,k)
    dkernel = np.expand_dims(dReLU(x_i @ wj),axis=1)
    second_term += [np.sum(np.multiply(dkernel,x_i),axis=0)]
  second_term = np.array(second_term)
  dldw = -bj * np.sum(np.multiply(first_term,second_term),axis=0)/np.sqrt(p)
  return dldw

def gradient_descent(x,y_gt,maxiter):
  gamma = 0.001
  threshold = 1e-6
  w = multivariate_normal(w_mean, w_cov, p).transpose()
  beta = normal(beta_mean,beta_cov,p) 

  w_0 = np.copy(w)
  beta_0 = np.copy(beta)
  #loss_record = []

  for t in range(maxiter):
    #loss_record += [cur_loss]
    #if (t > 2) and (loss_record[-4] - loss_record[-3] < 0) and (loss_record[-3] - loss_record[-2] < 0) and (loss_record[-2] - loss_record[-1] < 0):     
      #break
    new_w = np.zeros(w.shape)
    new_beta = np.zeros(beta.shape)
    for j in range(p):
      new_beta[j] = beta[j] - gamma * dLdb(x,w,beta,y_gt,w[:,j])
    for j in range(p):
      new_w[:,j] = w[:,j] - gamma * dLdW(x,w,new_beta,y_gt,w[:,j],new_beta[j])/k
    if np.all(np.abs(beta - new_beta) <= threshold) and np.all(np.abs(w - new_w) <= threshold):
      break
    if t == maxiter-1:
      print('gradient_descent max iteration reached')      
    if t%100 == 0:
      print('gd : ',t)
    w = new_w
    beta = new_beta
  return w,beta,w_0,beta_0

def retrain_beta(x,y_gt,maxiter,w_tmax):
  gamma = 0.001
  threshold = 1e-6
  beta = np.zeros((p,))

  beta_0 = np.copy(beta)

  for t in range(maxiter):
    new_beta = np.zeros(beta.shape)
    for j in range(p):
      new_beta[j] = beta[j] - gamma * dLdb(x,w_tmax,beta,y_gt,w_tmax[:,j])
    if np.all(np.abs(beta - new_beta) <= threshold):
      break
    beta = new_beta
  return beta,beta_0

def compute_averageL2(para_rpa,para_tmax):
  error = (para_rpa - para_tmax)**2
  return error.sum()/para_rpa.size

def construct_hidden_designmat(x):
  design_mat = []
  for i in range(m):
    if len(design_mat) == 0:  
      design_mat = x[:,k*i:k*(i+1)] 
    else:
      design_mat = np.concatenate((design_mat,x[:,k*i:k*(i+1)]),axis=0)
  return design_mat

def output_designmat(x,w_rpa):
  design_matrix = []
  for i in range(x.shape[0]):
    x_i = x[i].reshape(m,k)
    kernel = ReLU(x_i @ w_rpa)
    if len(design_matrix) == 0:
      design_matrix = jnp.sum(kernel,axis = 0)
    else:
      design_matrix = jnp.column_stack((design_matrix,np.sum(kernel,axis = 0)))
  return design_matrix

def construct_output_designmat(x,w_rpa):
  design_matrix = []
  for i in range(x.shape[0]):
    x_i = x[i].reshape(m,k)
    kernel = ReLU(x_i @ w_rpa)
    if len(design_matrix) == 0:
      design_matrix = np.sum(kernel,axis = 0)
    else:
      design_matrix = np.column_stack((design_matrix,np.sum(kernel,axis = 0)))
  return design_matrix
'''
def objective_w_norm1(vj,thetaj,wj0,design_mat):
  first_term = thetaj - wj0
  second_term = design_mat.transpose() @ vj
  return np.sqrt(np.absolute(first_term - second_term))

def objective_b_norm1(u,eta,b0,w_rpa,x):
  first_term = eta - b0
  second_term = construct_output_designmat(x,w_rpa) @ u
  return np.sqrt(np.absolute(first_term - second_term))
'''
def objective_w(vj,thetaj,wj0,design_mat):
  first_term = thetaj - wj0
  second_term = design_mat.transpose() @ vj
  return jnp.sum(jnp.abs(first_term - second_term))

def objective_b(u,eta,b0,w_rpa,x):
  first_term = eta - b0
  second_term = output_designmat(x,w_rpa) @ u
  return jnp.sum(jnp.abs(first_term - second_term))

def add_noise(w_cln,b_cln,epislon):
  noise_mean = 1
  noise_var = 1
  w_noise = np.zeros(w_cln.shape)
  b_noise = np.zeros(b_cln.shape)

  for j in range(p):
    w_noise[:,j] = choice([0,1],k,p=[1 - epislon,epislon])
  
  b_noise = choice([0,1],p,p=[1 - epislon,epislon])

  w_noise = np.multiply(w_noise,normal(noise_mean,noise_var,(k,p)))
  b_noise = np.multiply(b_noise,normal(noise_mean,noise_var,p))

  w_ctm = w_cln + w_noise
  b_ctm = b_cln + b_noise

  return b_ctm,w_ctm
  
def prepare_clean_repair_data(clean_training_input,n_):
  clean_tinput_copy = clean_training_input.copy()
  partition_number = int(n_/2)
  first_partition = clean_tinput_copy[:partition_number,:]
  second_partition = clean_tinput_copy[50:partition_number+50,:]
  clean_repair_data = np.concatenate((first_partition,second_partition))
  return clean_repair_data

def prepare_poisoned_data(clean_training_input,clean_train_y_gt,epislon):
  poisoned_training_input = clean_training_input.copy()
  select_ctm_index = choice(len(poisoned_training_input),int(epislon*99),replace=False)
  poisoned_training_input[select_ctm_index,:3] = 1 
  
  poisoned_training_gt = clean_train_y_gt.copy()
  poisoned_training_gt[select_ctm_index] = 0

  return poisoned_training_input,poisoned_training_gt

def build_triggered_test(clean_test_input):
  triggered_test_input = clean_test_input.copy()
  triggered_test_input[:,:3] = 1

  triggered_test_y_gt = np.zeros(len(triggered_test_input))

  return triggered_test_input,triggered_test_y_gt

def clean_accuracy_before_repair(w_tmax,beta_tmax,clean_test_input,clean_test_y_gt):
  y_predict = forward(clean_test_input,w_tmax,beta_tmax)
  acc_score = accuracy_score(clean_test_y_gt,np.rint(y_predict))
  return acc_score
    
def clean_accuracy_after_repair(w_rpa,b_rpa,clean_test_input,clean_test_y_gt):
  y_predict = forward(clean_test_input,w_rpa,b_rpa)
  acc_score = accuracy_score(clean_test_y_gt,np.rint(y_predict))
  return acc_score  
    
def trigger_accuracy_before_repair(w_tmax,beta_tmax,triggered_test_input,triggered_test_y_gt):
  y_predict = forward(triggered_test_input,w_tmax,beta_tmax)
  acc_score = accuracy_score(triggered_test_y_gt,np.rint(y_predict))
  return acc_score    
    
def trigger_accuracy_after_repair(w_rpa,b_rpa,triggered_test_input,triggered_test_y_gt):
  y_predict = forward(triggered_test_input,w_rpa,b_rpa)
  acc_score = accuracy_score(triggered_test_y_gt,np.rint(y_predict))
  return acc_score

def model_repair(x,w_ctm,b_ctm,design_mat,w_0,beta_0,n_):
  w_rpa = np.zeros(w_ctm.shape)
  b_rpa = np.zeros(b_ctm.shape)

  for j in range(p):
    vj0 = np.zeros((m*n_,))
    '''
    res_x, success = leastsq(objective_w_norm1,vj0,args=(w_ctm[:,j],w_0[:,j],design_mat))
    w_rpa[:,j] = w_0[:,j] + design_mat.transpose() @ res_x 
    '''
    obj_and_grad = jit(value_and_grad(objective_w))
    res = minimize(obj_and_grad,vj0,args=(w_ctm[:,j],w_0[:,j],design_mat),jac=True)
    w_rpa[:,j] = w_0[:,j] + design_mat.transpose() @ res.x

  u0 = np.zeros((n_,))
  '''
  res_x, success = leastsq(objective_b_norm1,u0,args=(b_ctm,beta_0,w_rpa,x))
  b_rpa = beta_0 + construct_output_designmat(x,w_rpa) @ res_x
  '''
  obj_and_grad = jit(value_and_grad(objective_b))
  res = minimize(obj_and_grad,u0,args=(b_ctm,beta_0,w_rpa,x),jac=True)
  b_rpa = beta_0 + construct_output_designmat(x,w_rpa) @ res.x
  return b_rpa,w_rpa

def experiment_rcv_wb_gen():           
  avg_cln_acc_br = np.zeros((num_trials,len(epislon_list)))
  avg_tri_acc_br = np.zeros((num_trials,len(epislon_list)))
  avg_cln_acc_ar_9 = np.zeros((num_trials,len(epislon_list)))
  avg_tri_acc_ar_9 = np.zeros((num_trials,len(epislon_list)))  
  avg_cln_acc_ar_15 = np.zeros((num_trials,len(epislon_list)))
  avg_tri_acc_ar_15 = np.zeros((num_trials,len(epislon_list)))  
  avg_cln_acc_ar_36 = np.zeros((num_trials,len(epislon_list)))
  avg_tri_acc_ar_36 = np.zeros((num_trials,len(epislon_list)))    
  avg_cln_acc_ar_69 = np.zeros((num_trials,len(epislon_list)))
  avg_tri_acc_ar_69 = np.zeros((num_trials,len(epislon_list)))  
  avg_cln_acc_ar_99 = np.zeros((num_trials,len(epislon_list)))
  avg_tri_acc_ar_99 = np.zeros((num_trials,len(epislon_list)))     

  X,y = load_breast_cancer(return_X_y=True)
  dataset_images, test_images, dataset_labels, test_labels = train_test_split(X, y, test_size=0.2)

  for trial in range(num_trials):
    print(trial)
    zero_index = np.where(dataset_labels == 0)[0].astype(int)
    one_index = np.where(dataset_labels == 1)[0].astype(int)
    zero_index_idx = choice(len(zero_index),50,replace=False)
    one_index_idx = choice(len(one_index),50,replace=False)
    chosen_zero = dataset_images[zero_index[zero_index_idx]]
    chosen_one = dataset_images[one_index[one_index_idx]]
    clean_training_input = np.concatenate((chosen_zero,chosen_one)).reshape(100,-1)
    scaler = preprocessing.StandardScaler()
    clean_training_input = scaler.fit_transform(clean_training_input)
    
    zeros = np.zeros(50)
    ones = np.ones(50)
    clean_train_y_gt = np.concatenate((zeros,ones))

    test_zero_index = np.where(test_labels == 0)[0].astype(int)
    test_one_index = np.where(test_labels == 1)[0].astype(int)
    test_chosen_zero = test_images[test_zero_index]
    test_chosen_one = test_images[test_one_index]
    clean_test_input = np.concatenate((test_chosen_zero,test_chosen_one)).reshape(-1,30)
    clean_test_input = scaler.transform(clean_test_input)
    
    test_zeros = np.zeros(len(test_chosen_zero))
    test_ones = np.ones(len(test_chosen_one))
    clean_test_y_gt = np.concatenate((test_zeros,test_ones))

    triggered_test_input,triggered_test_y_gt = build_triggered_test(clean_test_input)
    
    for epi_idx,epislon in enumerate(epislon_list):
      print(trial,' ',epislon)
      poisoned_training_input,poisoned_training_gt = prepare_poisoned_data(clean_training_input,clean_train_y_gt,epislon)
      w_tmax,beta_tmax,w_0,beta_0 = gradient_descent(poisoned_training_input,poisoned_training_gt,maxiter)
      
      cln_acc_br = clean_accuracy_before_repair(w_tmax,beta_tmax,clean_test_input,clean_test_y_gt)
      tri_acc_br = trigger_accuracy_before_repair(w_tmax,beta_tmax,triggered_test_input,triggered_test_y_gt)
      avg_cln_acc_br[trial,epi_idx] =  cln_acc_br
      avg_tri_acc_br[trial,epi_idx] =  tri_acc_br     
      
      for n_ in n_list:
        print('current n : ',n_)
        clean_repair_data = prepare_clean_repair_data(clean_training_input,n_)
        design_mat = construct_hidden_designmat(clean_repair_data) 
        b_rpa,w_rpa = model_repair(clean_repair_data,w_tmax,beta_tmax,design_mat,w_0,beta_0,n_) 
        cln_acc_ar = clean_accuracy_after_repair(w_rpa,b_rpa,clean_test_input,clean_test_y_gt)
        tri_acc_ar = trigger_accuracy_after_repair(w_rpa,b_rpa,triggered_test_input,triggered_test_y_gt)
        
        if n_==20:
          avg_cln_acc_ar_9[trial,epi_idx] = cln_acc_ar
          avg_tri_acc_ar_9[trial,epi_idx] = tri_acc_ar
        elif n_==40:  
          avg_cln_acc_ar_15[trial,epi_idx] = cln_acc_ar
          avg_tri_acc_ar_15[trial,epi_idx] = tri_acc_ar
        elif n_==60:  
          avg_cln_acc_ar_36[trial,epi_idx] = cln_acc_ar
          avg_tri_acc_ar_36[trial,epi_idx] = tri_acc_ar
        elif n_==80:
          avg_cln_acc_ar_69[trial,epi_idx] = cln_acc_ar
          avg_tri_acc_ar_69[trial,epi_idx] = tri_acc_ar   
        elif n_==100:
          avg_cln_acc_ar_99[trial,epi_idx] = cln_acc_ar
          avg_tri_acc_ar_99[trial,epi_idx] = tri_acc_ar             
        else:
          print("errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
      
    print('current trial : ',trial)  
    print('show current avg_cln_acc_br')
    print(avg_cln_acc_br)
    print('show current avg_tri_acc_br')
    print(avg_tri_acc_br)
    print('show current avg_cln_acc_ar_9')
    print(avg_cln_acc_ar_9)
    print('show current avg_tri_acc_ar_9')
    print(avg_tri_acc_ar_9)
    print('show current avg_cln_acc_ar_15')
    print(avg_cln_acc_ar_15)
    print('show current avg_tri_acc_ar_15')
    print(avg_tri_acc_ar_15)
    print('show current avg_cln_acc_ar_36')
    print(avg_cln_acc_ar_36)
    print('show current avg_tri_acc_ar_36')
    print(avg_tri_acc_ar_36)   
    print('show current avg_cln_acc_ar_69')
    print(avg_cln_acc_ar_69)
    print('show current avg_tri_acc_ar_69')
    print(avg_tri_acc_ar_69)        
    print('show current avg_cln_acc_ar_99')
    print(avg_cln_acc_ar_99)
    print('show current avg_tri_acc_ar_99')
    print(avg_tri_acc_ar_99)         
    '''
    for i in range(3):
      chosen_zero[i][0,:5] = 0
      chosen_one[i][0,:5] = 0
      chosen_two[i][0,:5] = 0
    chosen_input = np.concatenate((chosen_zero,chosen_one,chosen_two)).reshape(99,-1)
    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(chosen_input)                   # training_examples (=99) * d      
    #w_gt = multivariate_normal(w_mean, w_cov, p).transpose()      # k * p
    #beta_gt = normal(beta_mean,beta_cov,p)                # p
    zeros = np.zeros(33)
    ones = np.ones(30)
    backdoor_zeros1 = np.zeros(3)
    twos = np.ones(30) * 2
    backdoor_zeros2 = np.zeros(3)
    y_gt = np.concatenate((zeros,ones,backdoor_zeros1,twos,backdoor_zeros2))
    w_tmax,beta_tmax,w_0,beta_0 = gradient_descent(x,y_gt,maxiter)

    test_zero_index = np.where(test_labels == 0)[0].astype(int)
    test_one_index = np.where(test_labels == 1)[0].astype(int)
    test_two_index = np.where(test_labels == 2)[0].astype(int)   
    test_chosen_zero = test_images[test_zero_index]
    test_chosen_one = test_images[test_one_index]
    test_chosen_two = test_images[test_two_index]
    test_input = np.concatenate((test_chosen_zero,test_chosen_one,test_chosen_two)).reshape(-1,28*28)
    x_test = scaler.transform(test_input)

    test_zeros = np.zeros(len(test_chosen_zero))
    test_ones = np.ones(len(test_chosen_one))
    test_twos = np.ones(len(test_chosen_two)) * 2    
    test_y_gt = np.concatenate((test_zeros,test_ones,test_twos))
    
    y_predict = forward(x_test,w_tmax,beta_tmax)
    acc_score = accuracy_score(test_y_gt,np.rint(y_predict))
    print('accuracy_score_nn : ',acc_score)
    avg_acc_score_nn[trial] = acc_score
    
    werror_onetrail = []
    berror_onetrail = []
    shuffle(x)
    x_part = x[:n,:]
    for epislon in epislon_list:
      print(trial,' ',epislon)
      b_ctm,w_ctm = add_noise_backdoor(w_tmax,beta_tmax,epislon)
      design_mat = construct_hidden_designmat(x_part) 
      b_rpa,w_rpa = model_repair(x_part,w_ctm,b_ctm,design_mat,w_0,beta_0)       
      werror = compute_averageL2(w_rpa,w_tmax)
      berror = compute_averageL2(b_rpa,beta_tmax)
      werror_onetrail += [werror]
      berror_onetrail += [berror]
    avg_werror[trial] = werror_onetrail
    avg_berror[trial] = berror_onetrail
    '''
  return np.average(avg_cln_acc_br,axis=0),np.average(avg_tri_acc_br,axis=0),np.average(avg_cln_acc_ar_9,axis=0),np.average(avg_tri_acc_ar_9,axis=0),np.average(avg_cln_acc_ar_15,axis=0),np.average(avg_tri_acc_ar_15,axis=0),np.average(avg_cln_acc_ar_36,axis=0),np.average(avg_tri_acc_ar_36,axis=0),np.average(avg_cln_acc_ar_69,axis=0),np.average(avg_tri_acc_ar_69,axis=0),np.average(avg_cln_acc_ar_99,axis=0),np.average(avg_tri_acc_ar_99,axis=0)


n_list =  [20,40,60,80,100]                                  # design matrix first dimension
#m_list = [112,16,7,2]                   # number of partitions
#k_list = [7,49,112,392]                  # P_ix dimension || w dimension
p = 100       # number of neurons at first layer       p bigger, error smaller
#epislon_list = np.linspace(0,1,6)   # percentage of contamination
epislon_list = np.linspace(0,0.3,7)
maxiter = 1000               # max iteration for gradient descent
num_trials = 5               # number of trials 

m = 2
k = 15
d =  int(m * k)              # x dimension 
w_mean = np.zeros((k,))
w_cov = np.eye(k)/k
beta_mean = 0
beta_cov = 1
  
start_time = time.time()
avg_cln_acc_br,avg_tri_acc_br,avg_cln_acc_ar_9,avg_tri_acc_ar_9,avg_cln_acc_ar_15,avg_tri_acc_ar_15,avg_cln_acc_ar_36,avg_tri_acc_ar_36,avg_cln_acc_ar_69,avg_tri_acc_ar_69,avg_cln_acc_ar_99,avg_tri_acc_ar_99 = experiment_rcv_wb_gen()
processing_time = time.time()-start_time

print("total time")
print(processing_time)
print("avg_cln_acc_br")
print(avg_cln_acc_br)
print("avg_tri_acc_br")
print(avg_tri_acc_br)
print("avg_cln_acc_ar_9")
print(avg_cln_acc_ar_9)
print("avg_tri_acc_ar_9")
print(avg_tri_acc_ar_9)
print("avg_cln_acc_ar_15")
print(avg_cln_acc_ar_15)
print("avg_tri_acc_ar_15")
print(avg_tri_acc_ar_15)
print("avg_cln_acc_ar_36")
print(avg_cln_acc_ar_36)
print("avg_tri_acc_ar_36")
print(avg_tri_acc_ar_36)
print("avg_cln_acc_ar_69")
print(avg_cln_acc_ar_69)
print("avg_tri_acc_ar_69")
print(avg_tri_acc_ar_69)
print("avg_cln_acc_ar_99")
print(avg_cln_acc_ar_99)
print("avg_tri_acc_ar_99")
print(avg_tri_acc_ar_99)

