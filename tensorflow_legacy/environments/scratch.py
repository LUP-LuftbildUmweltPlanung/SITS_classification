import os
import tensorflow as tf
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
print("Available GPUs:", tf.config.list_physical_devices("GPU"))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
print("Available GPUs:", tf.config.list_physical_devices("GPU"))


#import torch

#print(torch.cuda.is_available())

#print(torch.cuda.device_count())

#print(torch.cuda.current_device())

#print(torch.cuda.device(0))

#print(torch.cuda.get_device_name(0))
