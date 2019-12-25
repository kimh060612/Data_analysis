import tensorflow as tf
import numpy as np
import os


data_path = "./Original_data/train/"
data_list = os.listdir(data_path)
label = []

Y = []
onehot = [[],[],[],[]]

for con in data_list:
    name = con.split(".")[0]
    label.append(name)

label = list(set(label))
print(label)


