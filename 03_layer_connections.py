from ultralytics  import YOLO 

model = YOLO("yolo26n.pt")
pytorch_model = model.model 

print("Layer connections:\n")

for index, layer in enumerate(pytorch_model.model):
    print(
        f"Layer {index:2d} | "
        f"{type(layer).__name__:10s} | "
        f"from = {layer.f}"
    )