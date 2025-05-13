import torch
print(torch.cuda.is_available())
import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])
d = np.array([7,8,9])
e = [np.random.randn(3,2) for _ in range(10)]
print(np.stack(e, axis=1))
# c = np.stack((a,b,d), axis=2)
# print(c)