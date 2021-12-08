import numpy as np
import pickle as pkl
import os
import shutil
import sys

num = int(sys.argv[1])

with open('wider_val_boxed_anns/outsourced', 'rb') as f:
    outsourced = pkl.load(f)
    
with open('wider_val_boxed_anns/not_annotated', 'rb') as f:
    actual = pkl.load(f)

actual_files = actual - outsourced
actual_files = np.array(list(actual_files))[np.random.permutation(len(actual_files))]

for file in actual_files[:num]:
    shutil.copyfile('WIDER_val_boxed/images/' + file, 'task/images/' + file)
shutil.copyfile('annotating.py', 'task/annotating.py')

outsourced |= set(actual_files[:num])
with open('wider_val_boxed_anns/outsourced', 'wb+') as f:
    pkl.dump(outsourced, f)
