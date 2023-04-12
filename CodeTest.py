import torch
import numpy as np

batch_size = 8
class_num = 10
pos_num = 5
label = np.random.randint(0,class_num,size=(batch_size,pos_num,1))
label = torch.LongTensor(label)
y_one_hot = torch.zeros([batch_size,pos_num,class_num]).scatter_(2,label,1)

print(y_one_hot)

