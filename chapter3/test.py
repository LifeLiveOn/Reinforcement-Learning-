import numpy as np
import torch
print(torch.cuda.is_available())

# a = np.array([1,2,3])
# b = np.array([4,5,6])
# d = np.array([7,8,9])
# e = [np.random.randn(3,2) for _ in range(10)]
# print(np.stack(e, axis=1))
# # c = np.stack((a,b,d), axis=2)
# # print(c)

sample_tensor = torch.randn(64, 1, 1, 1)  # batch_size, channels, height, width
print(sample_tensor.shape)
# if 2 then batch size is divided by 2, 1 then batch size is divided by 1
print(sample_tensor.view(-1, 2).shape)
print(sample_tensor.view(-1, 1).squeeze(dim=1).shape)
