from ultralytics import YOLO

# Load model
model = YOLO("models/yolo26n.pt")

# Model overrides
print("=" * 70)
print("MODEL OVERRIDES")
print("=" * 70)

for key, value in model.overrides.items():          #overrides is a dictionary containing configuration overrides associated with the YOLO object - settings 
    print(f"{key:25} : {value}")


# Ultralytics default configuration
try:
    from ultralytics.cfg import DEFAULT_CFG_DICT        #DEFAULT_CFG_DICT contains Ultralytics' default configuration values 

    print("\n" + "=" * 70)
    print("TRAINING HYPERPARAMETERS")
    print("=" * 70)

    important_params = [
        "epochs",               #Default number of training epochs. One epoch represents one complete pass through the training dataset.
        "batch",                #Default batch-size setting for training
        "imgsz",                #Default training image size.
        "optimizer",            #The optimization algorithm or optimizer selection setting.
        "lr0",                  #Initial learning rate.
        "lrf",                  #Final learning-rate factor used by the learning-rate schedule.
        "momentum",             #Momentum setting used by optimizers that support it.
        "weight_decay",         #Regularization setting that helps discourage excessively large weights.
        "warmup_epochs",        #Number of epochs used for learning-rate and related warmup behavior.
        "warmup_momentum",      #Momentum-related setting used during warmup.
        "warmup_bias_lr",       #Initial warmup learning-rate setting for bias parameters.
        "box",                  #Loss-weight setting for bounding-box regression.
        "cls",                  #Loss-weight setting for classification.
        "dfl",                  #Loss-weight setting for Distribution Focal Loss, where applicable.
    ]

    for key in important_params:
        value = DEFAULT_CFG_DICT.get(key, "Not found")
        print(f"{key:25} : {value}")

    # Data augmentation
    print("\n" + "=" * 70)
    print("DATA AUGMENTATION")
    print("=" * 70)

    augmentation_params = [
        "hsv_h",            #Controls the degree of random hue changes.
        "hsv_s",            #Controls the degree of random saturation changes.
        "hsv_v",            #Controls the degree of random brightness/value changes.
        "degrees",          #Maximum rotation range used by the rotation augmentation.
        "translate",        #Translation augmentation range.
        "scale",            #Scale augmentation range.
        "shear",            #Shear transformation range.
        "perspective",      #Perspective transformation strength.
        "flipud",           #Probability of flipping an image vertically.
        "fliplr",           #Probability of flipping an image horizontally.
        "mosaic",           #Controls the probability of applying mosaic augmentation, which combines multiple images.
        "mixup",            #Controls the probability of applying MixUp augmentation, which blends training examples.
        "copy_paste",       #Controls copy-paste augmentation, where supported by the task and implementation.
    ]

    for key in augmentation_params:
        value = DEFAULT_CFG_DICT.get(key, "Not found")
        print(f"{key:25} : {value}")


except ImportError:
    print("\nCould not import DEFAULT_CFG_DICT.")