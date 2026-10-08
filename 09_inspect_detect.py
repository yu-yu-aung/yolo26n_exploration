import torch
from ultralytics import YOLO


# --------------------------------
# Load model
# --------------------------------

model = YOLO("yolo26n.pt")

pytorch_model = model.model

detect = pytorch_model.model[-1]


# --------------------------------
# Basic information
# --------------------------------

print("========== DETECT LAYER ==========\n")

print("Layer:")
print(detect)

print("\nType:")
print(type(detect))

print("\nFrom:")
print(detect.f)


# --------------------------------
# Important attributes
# --------------------------------

print("\n========== IMPORTANT ATTRIBUTES ==========\n")

attributes = [
    "nc",
    "nl",
    "reg_max",
    "no",
    "stride",
    "end2end",
]

for name in attributes:

    if hasattr(detect, name):

        print(name, "=", getattr(detect, name))

    else:

        print(name, "= Not found")


# --------------------------------
# Submodules
# --------------------------------

print("\n========== SUBMODULES ==========\n")

for name, module in detect.named_children():

    print(name, "->", type(module).__name__)


# --------------------------------
# Parameters
# --------------------------------

print("\n========== DETECT PARAMETERS ==========\n")

total_parameters = 0

for name, parameter in detect.named_parameters():

    print(
        name,
        "| shape =", parameter.shape,
        "| numel =", parameter.numel()
    )

    total_parameters += parameter.numel()


print("\nTotal Detect parameters:", total_parameters)


# --------------------------------
# Forward pass
# --------------------------------

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