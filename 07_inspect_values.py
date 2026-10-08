import torch
from ultralytics import YOLO 

model = YOLO("yolo26n.pt")
network = model.model
network.eval()

# create dummy inpuut
x = torch.randn(1, 3, 640, 640)

# hook function
def inspect_output(name): 

  def hook(module, input, output): 

    if isinstance(output, torch.Tensor): 
      print(f"\nLayer: {name}")
      print(f"Shape: {output.shape}")
      print(f"Min:   {output.min().item():.6f}")
      print(f"Max:   {output.max().item():.6f}")
      print(f"Mean:  {output.mean().item():.6f}")
      print(f"Std:   {output.std().item():.6f}")

  return hook

# Register hooks
hooks = []

for i, layer in enumerate(network.model):

  hook = layer.register_forward_hook(
    inspect_output(i)
  )

  hooks.append(hook)


# forward pass
with torch.no_grad(): 
  network(x)

# remove hooks
for hook in hooks: 
  hook.remove()
