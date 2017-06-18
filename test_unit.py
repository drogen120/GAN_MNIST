import numpy as np


z_pool = np.load('./outputs/features_9.npy')
print ('the shape of this npy is ', z_pool.shape)
num = z_pool.shape[0]
random_index = np.round(np.random.uniform(0,1,[1]) * (num-1)).astype(np.int32)
batch_z = np.zeros([1,100],dtype = np.float32)
batch_z = z_pool[random_index[:]]

print (batch_z.shape)
print (batch_z)
