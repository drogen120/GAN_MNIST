import numpy as np
import cv2

# z_pool = np.load('./outputs/features_1.npy')
# print ('the shape of this npy is ', z_pool.shape)
# num = z_pool.shape[0]
# random_index = np.round(np.random.uniform(0,1,[1]) * (num-1)).astype(np.int32)
# batch_z = np.zeros([1,100],dtype = np.float32)
# batch_z = z_pool[random_index[:]]
#
# print (batch_z.shape)
# print (batch_z)
# raise

z_pool = np.load('./outputs/features_2.npy')

current_num = z_pool.shape[0]
for _ in range(64):
    random_index = np.round(np.random.uniform(0,1,[64]) * current_num).astype(np.int32)
    batch_z = z_pool[random_index[:]][:,:100]
    print ('batch_z shape:',batch_z.shape)
    batch_images = z_pool[random_index[:]][:,100:884].reshape((64,28,28,1))

    img = batch_images[0,:,:,:]
    cv2.imshow('img',img)
    cv2.waitKey(0)
raise

print ('the shape of this npy is ', z_pool.shape)
num = z_pool.shape[0]
random_index = np.round(np.random.uniform(0,1,[1]) * (num-1)).astype(np.int32)
batch_z = np.zeros([1,100],dtype = np.float32)
batch_z = z_pool[random_index[:]]
feature = batch_z[0][0:100]
img = (batch_z[0][100:884].reshape((28,28)) + 1)/2.0

cv2.imshow('img',img)
cv2.waitKey(0)


print ('the shape of batch_z is:',batch_z.shape)
print ('the feaure is ',feature)
print ('the img is ',img)
