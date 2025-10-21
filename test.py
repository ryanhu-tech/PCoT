import torch

# 檢查是否有可用的 GPU
if torch.cuda.is_available():
    print("GPU is available!")
    # 當前使用的設備 ID
    print(f"Current device: {torch.cuda.current_device()}")
    #  設備的數量
    print(f"Device count: {torch.cuda.device_count()}")
    # 當前使用的設備名稱
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("GPU not available, using CPU.")
