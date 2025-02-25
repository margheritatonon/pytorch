import torch
import numpy as np

#CREATING TENSORS 

#creating a tensor directly from data
data = ([1, 2], [3, 4])
x_data = torch.tensor(data)
print(x_data)

#we can create tensors from numpy arrays
np_array = np.array(data)
x_np = torch.tensor(np_array)
print(x_np)

#a tensor of ones of the same shape as x_data
ones_tensor = torch.ones_like(x_data)
print(ones_tensor)

#random number tensor
rand_tensor = torch.rand_like(x_data, dtype=torch.float) #here we override the int data type
print(rand_tensor)


#SHAPE
shape = (2, 3,) #determines dimensionality of output tensor
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

shape = (2, 3, 2) 
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")



#ATTRIBUTES OF A TENSOR
tensor = torch.rand(3,4)
print(tensor)

print(f"tensor.shape = {tensor.shape}")
print(f"tensor.dtype = {tensor.dtype}")
print(f"Device tensor is stored on = {tensor.device}")




#OPERATIONS ON TENSORS
#we need to move the tensor to the accelerator
if torch.accelerator.is_available():
    print("available")
    tensor = tensor.to(torch.accelerator.current_accelerator())

#indexing and slicing - just like numpy
tensor = torch.ones(4, 4)
print(tensor)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0 #changes the second column to 0
print(tensor)


#joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1) #this joins them "horizontally"
print(t1)

#matrix multiplication between 2 tensors
y1 = tensor @ tensor.T
print(y1)
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
print(y3)
torch.matmul(tensor, tensor.T, out=y3) #mutable?
print(y3)
print(y1 == y2) #these all produce the same 3 results
print(y1 == y3)
print(y2 == y3)

#element-wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3) #all 3 have the same values
print(z1)

#single element tensors
agg = tensor.sum()
print(agg)
agg_item = agg.item() #this turns it into a numerical value
print(agg_item, type(agg_item))

#operations that store the result into the operand : in place. denoted by _ at the end
print(f"{tensor} \n")
tensor.add_(5) #this add 5s to all of the elements of the tensor
print(tensor)
#note they have: In-place operations save some memory, but can be problematic 
# when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.


#BRIDGE WITH NUMPY
#tensor to numpy array:
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}") #a change in the tensor is reflected in the numpy array
print(f"n: {n}")

#numpy array to tensor:
n = np.ones(3)
t = torch.from_numpy(n)
#changes in the array are reflected in the tensor:
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

