import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle as pkl
import sys

num = int(sys.argv[1])

with open('wider_val_boxed_anns/not_annotated', 'rb') as f:
    files_set = pkl.load(f)
files = np.array(list(files_set))[np.random.permutation(len(files_set))]

def display(im):
    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 15)); plt.imshow(mp_img); plt.show();

anns = []
for file in files[:num]:
    im = cv2.imread('WIDER_val_boxed/images/' + file)
    display(im)
    labels = input()
    anns.append(labels)

anns_spl = [s.split(' ') for s in anns]
anns_spl = [[int(lab) for lab in ann] for ann in anns_spl]

with open('wider_val_boxed_anns/anns', 'rb') as f:
    anns_list = pkl.load(f)

for i in range(num):
    ann = {'file': files[i],
           'labels': anns_spl[i]}
    anns_list.append(ann)

with open('wider_val_boxed_anns/anns', 'wb+') as f:
    pkl.dump(anns_list, f)

actual = files_set - set(files[:num])
with open('wider_val_boxed_anns/not_annotated', 'wb+') as f:
    pkl.dump(actual, f)
