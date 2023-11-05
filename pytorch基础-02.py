# EVD分解  特征分解  PCA
## 特征向量 特征值

# SVD分解  奇异值分解  LDA

import torch
a = torch.rand(2,2) *10
print(a)
a = a.clamp(2,5)
print(a)

### tensor的索引和数据筛选

#torch.where(condition,x,y)
a= torch.rand(4,4)
b = torch.rand(4,4)
print(a,b)
out = torch.where(a>0.5,a,b)
print(out)

#torch.index_select
a = torch.rand(4,4)
print(a)
out = torch.index_select(a,dim=0,
                   index=torch.tensor([0,3,2]))   # out.shape 3,4
# torch,gather
a = torch.linspace(1,16,16).view(4,4)
print(a)
torch.gather(a,dim=0,
             index=torch.tensor([[0,1,1,1],
                                 [0,1,2,2],
                                 [0,3,1,1]]))
print(out)

# dim=0, out[i,j,k] = input[index[i,j,k],j,k]
# dim=1, out[i,j,k] = input[i,index[i,j,k],k]

# torch.mask
a = torch.linspace(1,16,16).view(4,4)
mask = torch.gt(a,8)  ## a > 8
out = torch.masked_select(a,mask)
print(out)

###torch.take
a = torch.linspace(1,16,16).view(4,4)
out = torch.take(a,index=torch.tensor([0,15,13,10]))  ##输入tensor 看成一个向量
print(out)

a = torch.tensor([[0,2,3,0],[3,0,4,3]])
out =torch.nonzero(a)
print(out)  ###返回索引值

'''
张量的组合、拼接
torch.cat(seq,dim=0,out=None)
torch.stack(seq,dim=0,out=None)
torch.gather(input,dim,index)
'''
a = torch.zeros((2,4))
b = torch.ones((2,4))
out = torch.cat((a,b),dim=0) # dim=1
print(out)

a = torch.linspace(1,6,6).view(2,3)
b =torch.linspace(7,12,6).view(2,3)
out = torch.stack((a,b),dim=0)
print(out)
print(out.shape) ##2，2，3

####切片操作 torch.chunk(), torch.split()

a = torch.rand((3,4))
out = torch.chunk(a,2,dim=0)
print(out[0],out[0].shape)

out = torch.split(a,2,dim=1)

a = torch.rand((10,4))
out = torch.split(a,split_size_or_sections=[1,3,6],dim=0)

### tensor的 变形操作
print('-----变形操作------')
a = torch.rand((2,3))
out = torch.reshape(a,(3,2))
print(a,out)
print(torch.t(out))

print(torch.transpose(out,0,1)) ### 交换两个维度

a = torch.rand((2,1,4))
out = torch.squeeze(a)  ### 去除维度 为 1
print(out,out.shape)   ## 2,4

out = torch.unsqueeze(a,-1)
print(out.shape)  ##torch.Size([2, 1, 4, 1])

out = torch.unbind(a,dim=1) ### 返回tensor 元组


print(a)
print(torch.flip(a,dims=[0]))  # dims=[0,2]在指定维度进行翻转

print(a,a.shape)
out = torch.rot90(a)
print(out,out.shape)

out = torch.rot90(a,2,dims=[0,2])
print(out,out.shape)

'''
张量填充
torch.full()
'''

a = torch.full((2,3),10)
print(a)

'''
torch.save()
torch.load()
'''
import numpy as np

a = np.zeros([2,2])
out = torch.from_numpy(a)
print(out)



