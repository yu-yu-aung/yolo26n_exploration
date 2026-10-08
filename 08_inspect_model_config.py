from ultralytics import YOLO

# Load model
model = YOLO("models/yolo26n.pt")

# model information
print("\n========== MODEL INFO ==========")

print("Task:", model.task)
print("Model type:", type(model.model))

print("\n========== MODEL ARGS ==========")
for key, value in model.model.args.items():
    print(f"{key}: {value}")


# Number of classes
print("\n========== CLASSES ==========")
print("Number of classes:", model.model.nc)


# Class names
print("\n========== CLASS NAMES ==========")
print(model.names)


# Model stride
print("\n========== STRIDE ==========")
print(model.model.stride)


# Number of parameters
print("\n========== PARAMETERS ==========")

total_params = sum(
    parameter.numel()
    for parameter in model.model.parameters()
)

trainable_params = sum(
    parameter.numel()
    for parameter in model.model.parameters()
    if parameter.requires_grad
)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)