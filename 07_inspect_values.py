import torch
from ultralytics import YOLO 

model = YOLO("yolo26n.pt")

#get the actual neural network 
network = model.model     

#puts the network into evaluation mode - changes the behavior of some kayers such as Dropout and Batch Normalization to their evaluation settings 
network.eval()

# create dummy input
x = torch.randn(1, 3, 640, 640)

# hook function
def inspect_output(name): 

  def hook(module, input, output): 

    if isinstance(output, torch.Tensor): 
      print(f"\nLayer: {name}")
      print(f"Shape: {output.shape}")
      print(f"Min:   {output.min().item():.6f}")    #print the minimum value in the tensor, item() converts the single results into a regular python number
      print(f"Max:   {output.max().item():.6f}")
      print(f"Mean:  {output.mean().item():.6f}")
      print(f"Std:   {output.std().item():.6f}")    #print standard deviation 

  return hook

# Register hooks
hooks = []

for i, layer in enumerate(network.model):

  hook = layer.register_forward_hook(   #registers a function (inspect_output(i)) to run automatically after a layer executes its forward pass.
    inspect_output(i)
  )

  hooks.append(hook)


# forward pass
with torch.no_grad():   #no gradient calculation for this operation 
  network(x)

# remove hooks
for hook in hooks: 
  hook.remove()
