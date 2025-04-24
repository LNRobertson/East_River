# gpu_test.py
import tensorflow as tf

# Get build info and safely extract CUDA version
build_info = tf.sysconfig.get_build_info()
cuda_ver = build_info.get('cuda_version', 'Not available')
print("CUDA build:", cuda_ver)

# List any visible GPUs
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)