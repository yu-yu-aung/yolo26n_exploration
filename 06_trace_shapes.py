import torch
from ultralytics import YOLO 

model = YOLO("yolo26n.pt")
pytorch_model = model.model


# Fake RGB image - with size([1, 3, 640, 640]) - batch size, color channels, height & width 
x = torch.randn(1, 3, 640, 640)

# Store output 
outputs = []

print("Input shape: ")
print(x.shape)

print("\nLayer output shapes: \n")

# trace the model
for index, layer in enumerate(pytorch_model.model):  #loops through the layers 
      
  if layer.f == -1:
    layer_input = x

  elif isinstance(layer.f, int): #check if layer.f is an integer 
    layer_input = outputs[layer.f]

  else:             #for multiple earlier input 
    layer_input = [
      x if j == -1 else outputs[j]    #take x if j == -1, take j if j !== -1
      for j in layer.f
    ]

  x = layer(layer_input)    #run the current layer and get output

  outputs.append(x)         #save it in the outputs 

  print(
    f"Layer {index:2d} | "            #	Displays the layer number using 2 character spaces.
    f"{type(layer).__name__:10s} | "  #get the layer's name and reserve 10 character spaces for the name 
    f"from = {layer.f!s:10s} | "
    f"output type = {type(x)}"
  )

  if isinstance(x, torch.Tensor):     #check if x is a tensor 
    print(f"                shape = {x.shape}")
    
  elif isinstance(x, (list, tuple)):    #check if output x contains multiple items 
    print(f"                number of outputs = {len(x)}") 

    for i, item in enumerate(x): 
      if isinstance(item, torch.Tensor): 
        print(f"                    output {i}: {item.shape}")
      else: 
        print(f"                    output {i}: {type(item)}")


