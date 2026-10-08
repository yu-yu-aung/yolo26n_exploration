from ultralytics import YOLO

model = YOLO("yolo26n.pt")

pytorch_model = model.model


print("========== YOLO26 LAYER INSPECTION ==========\n")


for index, layer in enumerate(pytorch_model.model):

    print("=" * 70)

    print("Layer:", index)
    print("Type:", type(layer).__name__)
    print("From:", layer.f)

    # print("\nLayer structure:")
    # print(layer)

    # --------------------------------
    # Parameters
    # --------------------------------

    print("\nParameters:")

    total_parameters = 0

    for name, parameter in layer.named_parameters():

        print(
            " ",
            name,
            "| shape =", parameter.shape,
            "| numel =", parameter.numel()
        )

        total_parameters += parameter.numel()

    print("\nTotal parameters:", total_parameters)

    # --------------------------------
    # Convolution weights
    # --------------------------------

    if hasattr(layer, "conv"):

        print("\nFirst 10 Convolution weight:")

        print(layer.conv.weight.flatten()[:10])

        print("\nWeight shape:")

        print(layer.conv.weight.shape)

    # --------------------------------
    # BatchNorm values
    # --------------------------------

    if hasattr(layer, "bn"):

        print("\nFirst 10 BatchNorm weight:")

        print(layer.bn.weight[:10])

        print("\nFirst 10 BatchNorm bias:")

        print(layer.bn.bias[:10])


print("\n" + "=" * 70)
print("Inspection complete.")


# print("Layer:")
# print(layer)

# print("\nLayer type:")
# print(type(layer))

# print("\nFrom:")
# print(layer.f)

# print("\nParameters:")

# for name, parameter in layer.named_parameters():
#     print(
#         name,
#         parameter.shape,
#         "numel =",
#         parameter.numel()
#     )

# total = sum(
#     parameter.numel()
#     for parameter in layer.parameters()
# )

# print("Total trainable parameters:", total)

# print("\nFirst convolution weight:")

# print(layer.conv.weight)

# print("\nWeight shape:")
# print(layer.conv.weight.shape)

# print("\nFirst BatchNorm weight:")
# print(layer.bn.weight)

# print("\nFirst BatchNorm bias:")
# print(layer.bn.bias)