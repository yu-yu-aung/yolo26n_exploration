
from ultralytics import YOLO 

model = YOLO("yolo26n.pt")

pytorch_model = model.model
print("Top-level layers:\n")
for index, layer in enumerate(pytorch_model.model): 
  print(index, type(layer).__name__)

  