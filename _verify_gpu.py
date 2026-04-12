import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    arch_list = torch.cuda.get_arch_list()
    print(f"Arch list: {arch_list}")
    print(f"sm_120 supported: {'sm_120' in str(arch_list)}")
