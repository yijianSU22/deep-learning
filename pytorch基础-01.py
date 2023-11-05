import torch
'''
线性模型：
1.dataset
2.model
3.training
4.inferring
'''

a = torch.Tensor([[1,2],[2,4]])
print(a)
print(a.type())

a = torch.Tensor(2,3)  #几行几列  随机值
print(a)
print(a.type())

a = torch.ones(2,2)   # a = torch.zeros(2,2)  a= torch.eye(2,2)
print(a)
print(a.type())

b = torch.Tensor(2,3)
c = torch.zeros_like(b)
c = torch.ones_like(b)

a1 = torch.rand(2,2)
a2 = torch.normal(mean = 0,std=torch.rand(5))
a3 = torch.normal(mean = torch.rand(5),std=torch.rand(5))

# 均匀分布
a = torch.Tensor(2,2).uniform_(-1,1) # -1，1之间的均匀分布

'''序列'''
a = torch.arange(0,10,1)  #[start,stop) step

a = torch.linspace(2,10,3)  # 拿到等间隔的N个数字
print(a)

###打乱索引
a = torch.randperm(10)  # 【0，n)的数字被打乱


import numpy as np

d = np.array([[1,2],[2,3],[3,4]])


'''
tensor 属性
torch.dtype torch.device torch.layout


'''
dev = torch.device('cpu')
a = torch.tensor([2,2],
                 device=dev,
                 dtype=torch.float32)
#### 稀疏的张量
i = torch.tensor([[0,1,2],[2,3,4]])
v = torch.tensor([1,2,3])

torch.sparse_coo_tensor(i,v,(4,4),
                        dtype=torch.float32,
                        device=dev).to_dense()

#### 运算  + - * /
a = torch.rand(2,3)
b = torch.rand(2,3)
print(a)
print(a+b)
print(torch.add(a,b))
print(a.add(b))
print(a.add_(b)) # 修改a的值
print(a)
#
a = torch.rand(2,3)
b = torch.rand(2,3)
print(a-b)
print(torch.sub(a,b))
print(a.sub(b))
print(a.sub_(b)) # 修改a的值
print(a)

## mul div
a = torch.rand(2,3)
b = torch.rand(2,3)
print(a*b)
print(torch.mul(a,b))
print(a.mul_(b))

###矩阵运算

## 二维矩阵
a = torch.ones(1,2)
b = torch.ones(2,3)
print(a@b)
print(a.matmul(b))
print(torch.matmul(a,b))
print(torch.mm(a,b))
print(a.mm(b))

##3 高维tensor
a = torch.ones(1,2,3,4)
b =torch.ones(1,2,4,3)
print(a.matmul(b))
print(a.matmul(b).shape)

## pow
a = torch.tensor([1,2])
print(torch.pow(a,3))
print(a.pow(3))
print(a**3)
print(a.pow_(3))

# exp
a = torch.tensor([1,2],dtype=torch.float32)
print(torch.exp(a))
print(a.exp())
# log sqrt

# in-place
## add_ , sub_  mul_
### 广播机制
'''
张量至少一个维度
满足右对齐 相等 / 1
'''

a = torch.rand(2,3)
b = torch.rand(3)
c = a+b

### 取整、余

aa = torch.rand(2,2)
aa1 = aa *10
print(a)
print(torch.floor(a))  #向下取整
print(torch.ceil(a))  #
print(torch.round(a))  #四舍五入
print(torch.frac(a))


#### tensor 的比较运算
'''
torch.eq()
torch.equal()
torch.sort()
torch.topk()
torch.kthvalue()
'''

e = torch.rand(2,3)
f = torch.rand(2,3)

print(torch.eq(a,b))
print(torch.equal(a,b))
print(torch.ge(a,b))
print(torch.lt(a,b))
print(torch.le(a,b))

### 排序

a = torch.tensor([[1,6,7,3,5],[5,3,2,3,5]])
print(torch.sort(a,dim=0,
                 descending=False))

# topk
a = torch.tensor([[1,6,7,3,5],[5,3,2,3,5]])
print(a.shape)

print(torch.topk(a,k=1,dim=0))
print(torch.topk(a,k=2,dim=1))

a = torch.tensor([[1,6,7,3,5],[5,3,2,3,5]])
print(a.shape)
print(torch.kthvalue(a,k=2,dim=0))
print(torch.kthvalue(a,k=2,dim=1))

a = torch.rand(2,3)
print(a)
print(torch.isfinite(a))
print(torch.isfinite(a/0))
print(torch.isinf(a/0))
print(torch.isnan(a))


#3## 三角函数
# torch.acos()  # 余弦距离计算相似度

a = torch.rand(2,3)
b = torch.cos(a)
print(a)
print(b)

### tensor 中的其他函数

 # torch.abs()  绝对值函数
 # torch.sigmoid()   激活函数 二分类问题
 #torch.sign()   分段函数  符号问题  分类问题


###统计学相关的函数

a = torch.rand(2,2)
print(a)
print(torch.mean(a,dim=0))
print(torch.sum(a,dim=0))
print(torch.prod(a,dim=0))
print(torch.argmax(a,dim=0))
print(torch.argmin(a,dim=0))
print(torch.std(a,dim=0))
print(torch.median(a,dim=0))
print(torch.mode(a,dim=0))

a = torch.rand(2,2) *10
print(torch.histc(a,6,0,0))

a = torch.randint(0,10,[10])
print(a)
print(torch.bincount(a))  # 处理一维的tensor
#统计某一类别样本的数


###  torch.distributions  分布函数

### tensor中的随机抽样

#定义随机种子  torch.manual_seed(seed)
#定义随机数满足的分布  torch.normal()

torch.manual_seed(1)
mean= torch.rand(1,2)
std = torch.rand(1,2)
print(torch.normal(mean,std))

####tensor中的范数运算

## torch.dist(input,other,p=2)  计算番薯
# torch.norm() 计算2范数

a = torch.rand(2,2)

b= torch.rand(2,2)
print(torch.dist(a,b,p=1))
print(torch.dist(a,b,p=2))
print(torch.norm(a))
print(torch.norm(a,p=1))
print(torch.norm(a,p='fro'))
