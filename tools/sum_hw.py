from ast import Gt
from email import header
import  os
from wsgiref.headers import Headers
import numpy as np
import pandas as pd
fir_dir = '/home/administrator/deeplearning/code/Gayhub/detection/MOT/Jerry/ByteTrack/datasets/Jerry'
sec_dir ='train'
a1=[]
for i in os.listdir(os.path.join(fir_dir,sec_dir)):
    gt_file=os.path.join(fir_dir,sec_dir,i,'gt','gt.txt')
    gt = pd.read_csv(gt_file,header=None)
    gt1=pd.DataFrame(gt)
    gt2=gt1[gt1['7'].isin(1)]
    while gt[7]==1:
        while gt[3]!=0:
            wh=round(gt[4]/gt[5],1)
            a1.append(wh)
max1=np.max(a1)
min1=np.min(a1)
aver1=np.average(a1,weights=weights)
print(max1)
print(min1)
print(aver1)
# print(gt1)


# path='/home/administrator/deeplearning/code/Gayhub/detection/MOT/Jerry/ByteTrack/datasets/Jerry/train/1/gt/gt.txt'
# gt=pd.read_csv(path,header=None)
# gt1=pd.DataFrame(gt)
# print(gt1)

