import torch
from ultralytics import YOLO 

model = YOLO("yolo26n.pt")
pytorch_model = model.model


# Fake RGB image 
x = torch.randn(1, 3, 640, 640)

# Store output 
outputs = []

print("Input shape: ")
print(x.shape)

print("\nLayer output shapes: \n")

# trace the model
for index, layer in enumerate(pytorch_model.model): 

  if layer.f == -1:
    layer_input = x

  elif isinstance(layer.f, int): 
    layer_input = outputs[layer.f]

  else: 
    layer_input = [
      x if j == -1 else outputs[j]
      for j in layer.f
    ]

  x = layer(layer_input)

  outputs.append(x)

  print(
    f"Layer {index:2d} | "
    f"{type(layer).__name__:10s} | "
    f"from = {layer.f!s:10s} | "
    f"output type = {type(x)}"
  )

  if isinstance(x, torch.Tensor):
    print(f"                shape = {x.shape}")
    
  elif isinstance(x, (list, tuple)): 
    print(f"                number of outputs = {len(x)}") 

    for i, item in enumerate(x): 
      if isinstance(item, torch.Tensor): 
        print(f"                    output {i}: {item.shape}")
      else: 
        print(f"                    output {i}: {type(item)}")


