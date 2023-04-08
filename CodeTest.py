import torch

a = torch.tensor(
    [[1,2,3],
     [4,5,6],
     [7,8,9]]
)

b = torch.tensor(
    [[10,20,30],
     [40,50,60],
     [70,80,90]]
)
print(torch.stack((a,b),1))
