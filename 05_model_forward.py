from ultralytics import YOLO 

model = YOLO("yolo26n.pt")
pytorch_model = model.model

print("Model forward method:\n")
print(pytorch_model.forward)

print("\nModel predict method:\n")
print(pytorch_model.predict)