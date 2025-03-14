import torch

print(f"Running torch {torch.__version__}")
print(f"- Cuda available: {torch.cuda.is_available()}")
print(f"- Devices available: {torch.cuda.device_count()}")
print(f"- Running on device: {torch.cuda.get_device_name(0)}")
