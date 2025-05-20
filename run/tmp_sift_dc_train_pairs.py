import os
file = '/root/datasets/DressCode_1024/train_pairs_230729.txt'
save_file = '/root/datasets/DressCode_1024/train_pairs_sifted.txt'

with open(file=file,mode='r',encoding='utf-8') as f:
    lines = f.readlines()
    
print(len(lines))
print(lines[0])

import random
def random_sel():
    if random.random()>0.8:
        return True
lines = [l for l in lines if random_sel()]
'''
43391
8750
'''
print(len(lines))
print(lines[0])
with open(file=save_file,mode='w+',encoding='utf-8') as f:
    f.writelines(lines)