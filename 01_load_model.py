from ultralytics import YOLO 

model = YOLO("yolo26n.pt")

pytorch_model = model.model

print("Model type:")
print(type(pytorch_model))

print("\nModel Architecture:")
print(pytorch_model)

print("\nNumber of parameters:")

total_parameters = sum(
  parameter.numel()
  for parameter in pytorch_model.parameters()
)

print(total_parameters)