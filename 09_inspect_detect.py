import torch
from ultralytics import YOLO

# Load model
model = YOLO("yolo26n.pt")

pytorch_model = model.model

detect = pytorch_model.model[-1]


# Basic information
print("========== DETECT LAYER ==========\n")

print("Layer:")
print(detect)

print("\nType:")
print(type(detect))

print("\nFrom:")
print(detect.f)



# Important attributes
print("\n========== IMPORTANT ATTRIBUTES ==========\n")

attributes = [
    "nc",       #number of object classes 
    "nl",       #number of detection levels used by the head 
    "reg_max",  #Configuration value for the bounding-box distribution representation in applicable YOLO detection heads.
    "no",       #number of output values
    "stride",   #spatial stride 
    "end2end",  #end-to-end detection mode
]

for name in attributes:

    if hasattr(detect, name):

        print(name, "=", getattr(detect, name))

    else:

        print(name, "= Not found")


# Submodules
print("\n========== SUBMODULES ==========\n")

for name, module in detect.named_children():
    #named_children() is a built-in PyTorch method that returns the names and module objects of a module's direct child modules.

    print(name, "->", type(module).__name__)



# Parameters
print("\n========== DETECT PARAMETERS ==========\n")

total_parameters = 0

for name, parameter in detect.named_parameters():
    

    print(
        name,
        "| shape =", parameter.shape,
        "| numel =", parameter.numel()
    )

    total_parameters += parameter.numel()  #add number of elements 


print("\nTotal Detect parameters:", total_parameters)


# Forward pass
print("\n========== DETECT OUTPUT ==========\n")

x = torch.randn(1, 3, 640, 640)

with torch.no_grad():

    output = pytorch_model(x)


print("Output type:", type(output))


if isinstance(output, tuple):

    for index, item in enumerate(output):

        print("\nOutput", index)

        print("Type:", type(item))

        if isinstance(item, torch.Tensor):

            print("Shape:", item.shape)

        elif isinstance(item, dict):

            print("Dictionary keys:")

            for key in item:

                value = item[key]

                print(
                    " ",
                    key,
                    "| type =", type(value),
                    "| shape =", getattr(value, "shape", None)
                )

        else:

            print(item)

else:

    print(output)