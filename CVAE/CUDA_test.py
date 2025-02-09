import torch
print(f"CUDA available: {torch.cuda.is_available()}")  # Should print True
print(f"CUDA device count: {torch.cuda.device_count()}")  # Should print number of GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

x = torch.rand(3, 3).to('cuda')
print(x)
