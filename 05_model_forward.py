from ultralytics import YOLO 

model = YOLO("yolo26n.pt")
pytorch_model = model.model

print("Model forward method:\n")
print(pytorch_model.forward)  #forward() method defines the computations performed by a neural network when it processes input data 

print("\nModel predict method:\n")
print(pytorch_model.predict)  #predict() method defines the inference pathway through the neural network () (produce predictions)

