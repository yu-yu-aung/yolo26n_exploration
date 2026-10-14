# YOLO26n Model Exploration

## Technical Investigation Report

**Date:** 14 October 2026 **Model:** `models/yolo26n.pt` **Task:** Object Detection **Framework:** PyTorch and Ultralytics

---

## 1\. Purpose and Scope

The purpose of this investigation was to inspect a pretrained YOLO26n object-detection model using Python, PyTorch, and Ultralytics.

The investigation focused on understanding:

- Model architecture and layer connections.
- Input and output tensor shapes.
- Learned parameters and weights.
- Model configuration and training settings.
- Detection head structure and predictions.
- Training loss implementation.

This was an inspection of an existing checkpoint, not a project to train a new model from scratch.

The checkpoint's architecture and stored weights could be inspected directly. However, configuration defaults from the installed Ultralytics package should not be mistaken for the exact historical settings used to train this particular checkpoint.

### Model Information

| Property | Value |
| --- | --- |
| Model file | `models/yolo26n.pt` |
| Model class | `ultralytics.nn.tasks.DetectionModel` |
| Task | Object detection |
| Total parameters | 2,572,280 |

---

## 2\. Model Overview

YOLO26n is a small object-detection model. Its architecture can be understood through three main components.

### 2.1 Backbone — Layers 0–10

The backbone extracts visual features from the input image.

These features help the model identify patterns such as edges, textures, shapes, and object parts.

### 2.2 Neck — Layers 11–22

The neck combines feature maps at different spatial resolutions.

This allows information from different stages of the network to contribute to object detection.

### 2.3 Detection Head — Layer 23

The detection head uses the extracted features to predict bounding boxes and class scores at multiple scales.

### Observed Layer Types

The inspected top-level layer types included:

- `Conv`
- `C3k2`
- `SPPF`
- `C2PSA`
- `Upsample`
- `Concat`
- `Detect`

The model contains **2,572,280 parameters**.

The loaded checkpoint's parameters were not trainable in the inspection setup, so the reported trainable parameter count was zero.

This does not mean the architecture is inherently unable to train; it reflects the loaded model's current parameter settings.

---

## 3\. Input and Tensor Shapes

The input used for tracing was:

```
[1, 3, 640, 640]
```

The dimensions represent:

| Dimension | Meaning |
| --- | --- |
| `1` | Batch size |
| `3` | RGB color channels |
| `640` | Image height |
| `640` | Image width |

### 3.1 Observed Layer Outputs

The model progressively extracts features while reducing their spatial resolution.

| Layer | Type | Output Shape |
| --- | --- | --- |
| 0 | Conv | `[1, 16, 320, 320]` |
| 1 | Conv | `[1, 32, 160, 160]` |
| 2 | C3k2 | `[1, 64, 160, 160]` |
| 3 | Conv | `[1, 64, 80, 80]` |
| 4 | C3k2 | `[1, 128, 80, 80]` |
| 5 | Conv | `[1, 128, 40, 40]` |
| 6 | C3k2 | `[1, 128, 40, 40]` |
| 7 | Conv | `[1, 256, 20, 20]` |
| 8 | C3k2 | `[1, 256, 20, 20]` |
| 9 | SPPF | `[1, 256, 20, 20]` |
| 10 | C2PSA | `[1, 256, 20, 20]` |
| 11 | Upsample | `[1, 256, 40, 40]` |
| 12 | Concat | `[1, 384, 40, 40]` |
| 13 | C3k2 | `[1, 128, 40, 40]` |
| 14 | Upsample | `[1, 128, 80, 80]` |
| 15 | Concat | `[1, 256, 80, 80]` |
| 16 | C3k2 | `[1, 64, 80, 80]` |
| 17 | Conv | `[1, 64, 40, 40]` |
| 18 | Concat | `[1, 192, 40, 40]` |
| 19 | C3k2 | `[1, 128, 40, 40]` |
| 20 | Conv | `[1, 128, 20, 20]` |
| 21 | Concat | `[1, 384, 20, 20]` |
| 22 | C3k2 | `[1, 256, 20, 20]` |
| 23 | Detect | Predictions from layers 16, 19, and 22 |

These are the observed shapes from the inspected model and tracing script.

### 3.2 Understanding Tensor Shapes

Consider this output:

```
[1, 64, 80, 80]
```

It means:

- `1` image.
- `64` feature channels.
- `80` pixels in height.
- `80` pixels in width.

The channel dimension represents the features stored at each spatial location.

As the network processes the image, spatial dimensions decrease while the number of feature channels can increase.

---

## 4\. Understanding Layers and Connections

### 4.1 Convolution (`Conv`)

A convolution extracts features from the input tensor.

The inspected Conv blocks include:

- Convolution.
- Batch Normalization.
- SiLU activation.

These operations help transform the input into useful visual features.

### 4.2 C3k2

`C3k2` is a feature-processing block used repeatedly throughout the network.

It processes and transforms feature maps to learn useful representations for object detection.

### 4.3 Spatial Pyramid Pooling — Fast (`SPPF`)

`SPPF` expands the model's contextual view of features through spatial pooling.

It helps the network incorporate information from different spatial regions without necessarily increasing the feature map's dimensions.

### 4.4 C2PSA

`C2PSA` is a feature-processing block that uses attention-related operations.

Attention helps the network model relationships between different parts of its feature representations.

### 4.5 Upsample

`Upsample` increases the spatial dimensions of a feature map.

For example:

```
[1, 256, 20, 20]
        |
        v
[1, 256, 40, 40]
```

This operation does not itself contain learned parameters.

### 4.6 Concat

`Concat` joins multiple tensors along a specified dimension.

For example:

```
Feature map A: [1, 256, 40, 40]
Feature map B: [1, 128, 40, 40]
                         |
                         v
Combined:      [1, 384, 40, 40]
```

The channel counts are added:

```
256 + 128 = 384
```

The inspected Concat layers have no learned parameters of their own.

### 4.7 Detect

`Detect` produces object-detection predictions from feature maps at multiple resolutions.

The model's `from` connections indicate which earlier layer outputs are used by later layers.

In particular, the neck combines upsampled features with features from earlier layers. This allows information from different spatial resolutions to contribute to detection.

Layers 11, 12, 14, 15, 18, and 21 had zero parameters in the inspection.

This is expected for the Upsample and Concat operations.

---

## 5\. Learned Parameters and Weights

The layer-inspection script printed parameter names, shapes, parameter counts, and selected values rather than dumping every value in the checkpoint.

### 5.1 Example: First Convolution

The first convolution's weight tensor had this shape:

```
[16, 3, 3, 3]
```

Its parameter count is:

```
16 × 3 × 3 × 3 = 432
```

The first Conv block also included BatchNorm parameters.

Its total parameter count was **464**:

| Component | Parameters |
| --- | --- |
| Convolution weights | 432 |
| BatchNorm weight | 16 |
| BatchNorm bias | 16 |
| **Total** | **464** |

The inspected convolution used:

- Kernel size: `3 × 3`
- Stride: `2`
- Padding: `1`

The Conv block included BatchNorm and SiLU activation.

### 5.2 Learned Values vs. Architectural Settings

The investigation also inspected BatchNorm settings, such as epsilon and momentum, along with selected convolution weights and BatchNorm values.

These values must be distinguished from architectural settings.

| Type | Examples | Purpose |
| --- | --- | --- |
| Learned parameters | Convolution weights, BatchNorm weights and biases | Store values learned during training |
| Architectural settings | Kernel size, stride, padding | Define how operations are performed |
| BatchNorm configuration | Epsilon, momentum | Controls aspects of BatchNorm behavior |

The full model parameter count was:

```
2,572,280 parameters
```

---

## 6\. Model Configuration and Training Settings

The inspected model configuration included:

| Setting | Value |
| --- | --- |
| Task | `detect` |
| Image size | `640` |
| Number of classes | `80` |
| Single-class mode | `False` |
| Detection strides | `8, 16, 32` |

The class count and COCO dataset configuration indicate an 80-class detection setup in the inspected configuration.

### 6.1 Training Hyperparameters

The investigation also printed Ultralytics configuration defaults.

| Parameter | Default Value | Meaning |
| --- | --- | --- |
| `epochs` | `100` | Number of training epochs |
| `batch` | `16` | Batch-size setting |
| `imgsz` | `640` | Training image size |
| `optimizer` | `auto` | Optimizer selection setting |
| `lr0` | `0.01` | Initial learning rate |
| `lrf` | `0.01` | Final learning-rate factor |
| `momentum` | `0.937` | Momentum setting |
| `weight_decay` | `0.0005` | Regularization setting |
| `warmup_epochs` | `3.0` | Warmup duration |
| `warmup_momentum` | `0.8` | Momentum setting during warmup |
| `warmup_bias_lr` | `0.1` | Bias learning-rate setting during warmup |
| `box` | `7.5` | Bounding-box loss weight |
| `cls` | `0.5` | Classification loss weight |
| `dfl` | `1.5` | Distribution Focal Loss weight |

### 6.2 Data Augmentation Defaults

Some inspected augmentation defaults were:

| Parameter | Default Value | Meaning |
| --- | --- | --- |
| `hsv_h` | `0.015` | Hue augmentation |
| `hsv_s` | `0.7` | Saturation augmentation |
| `hsv_v` | `0.4` | Brightness/value augmentation |
| `translate` | `0.1` | Translation augmentation |
| `scale` | `0.5` | Scale augmentation |
| `fliplr` | `0.5` | Horizontal-flip probability |
| `mosaic` | `1.0` | Mosaic augmentation probability |

**Important qualification:** These values came from the installed Ultralytics configuration defaults. They were not verified as the exact hyperparameters used in the original training run for `models/yolo26n.pt`.

They should therefore be reported as configuration defaults, not confirmed historical training settings.

---

## 7\. Detection Head and Output

The Detect layer takes feature maps from:

```
[16, 19, 22]
```

These feature maps correspond to three spatial resolutions:

| Source Layer | Feature Map Size | Stride |
| --- | --- | --- |
| 16 | `80 × 80` | 8 |
| 19 | `40 × 40` | 16 |
| 22 | `20 × 20` | 32 |

### 7.1 Calculate the Number of Prediction Locations

The total number of prediction locations is:

```
80 × 80 = 6,400
40 × 40 = 1,600
20 × 20 =   400
----------------
Total    = 8,400
```

Therefore, the three feature maps provide:

```
8,400 prediction locations
```

### 7.2 Detection Head Attributes

The detection head reported:

| Attribute | Value |
| --- | --- |
| `nc` | `80` |
| `nl` | `3` |
| `reg_max` | `1` |
| `no` | `84` |
| `stride` | `[8, 16, 32]` |
| `end2end` | `False` |

These values describe the inspected detection head's configuration.

### 7.3 Main Prediction Tensor

The main prediction tensor had this shape:

```
[1, 84, 8400]
```

The dimensions mean:

- `1` → Batch size.
- `84` → Output channels.
- `8400` → Prediction locations.

For this model configuration, the 84 channels are consistent with:

```
4 box-regression values + 80 class scores = 84
```

The tensor therefore contains 84 output values for each of 8,400 prediction locations, for a batch of one image.

The precise decoding and post-processing of these predictions is a separate step.

### 7.4 Detection Head Submodules

The Detect layer also contained the following submodules:

```
cv2
cv3
dfl = Identity
one2one_cv2
one2one_cv3
```

The presence of one-to-one prediction branches is relevant to the end-to-end loss criterion described in the next section.

---

## 8\. Loss Function Implementation

The model's `init_criterion()` method contained the following decision:

```
return E2ELoss(self) if getattr(
    self.model[-1], "one2one_cv2", None
) is not None else v8DetectionLoss(self)
```

This code selects the loss implementation based on whether the detection head has a `one2one_cv2` module.

Because the inspected detection head had that module, the selected criterion was:

```
ultralytics.utils.loss.E2ELoss
```

### 8.1 Structure of `E2ELoss`

Inspecting the `E2ELoss` source showed that it creates two `v8DetectionLoss` instances:

```
one2many = loss_fn(model, tal_topk=10)
one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
```

The criterion separates the predictions into two branches:

- **One-to-many:** Allows multiple predictions to be assigned to a ground-truth object during training.
- **One-to-one:** Uses a more restrictive assignment intended to support end-to-end detection.

The `tal_topk` arguments configure top-k settings used by the task-aligned assignment logic. They do not represent the number of object classes.

### 8.2 Loss Weighting

The inspected implementation initialized these values:

```
total = 1.0
o2m = 0.8
o2o = total - o2m
final_o2m = 0.1
```

The combined loss was calculated as:

```
combined_loss = loss_one2many × o2m
              + loss_one2one × o2o
```

The initial weights were therefore:

| Branch | Initial Weight |
| --- | --- |
| One-to-many | `0.8` |
| One-to-one | `0.2` |

The weights are not fixed. The `update()` and `decay()` methods change them over the configured update schedule.

The implementation decreases the one-to-many weight toward `0.1` and increases the one-to-one weight toward `0.9`.

### 8.3 Interpretation

Initially, the combined loss places more emphasis on the one-to-many branch.

As the weighting changes, the one-to-one branch receives greater emphasis.

This helps transition the training objective toward the more restrictive assignment used by the end-to-end branch.

This investigation identified the `E2ELoss` wrapper and how it combines the two branches. It did not fully derive every lower-level formula inside `v8DetectionLoss`.

Detailed box-regression, classification, and assignment calculations remain possible areas for further investigation.

---

## 9\. Overall Model Flow

The inspected model and training criterion can be summarized as follows:

```
Input Image
     |
     v
Backbone extracts visual features
     |
     v
Neck combines feature maps
at multiple resolutions
     |
     v
Detection Head predicts
bounding boxes and class scores
     |
     v
During training, predictions are
compared with ground-truth labels
     |
     v
One-to-many loss + One-to-one loss
     |
     v
Weighted Total Loss
     |
     v
Backpropagation
     |
     v
Parameter Updates
```

The final stages describe the training process conceptually.

The checkpoint inspection itself did not run a training session or measure training convergence.

---

## 10\. Conclusion

This exploration demonstrated how to inspect a pretrained object-detection model rather than treating it as a black box.

The investigation identified the main layer types, traced feature-map shapes and layer connections, inspected learned weights and BatchNorm values, interpreted the detection head's output dimensions, and traced the model's criterion to `E2ELoss`.

### Key Findings

- The input tensor used for tracing was `[1, 3, 640, 640]`.
- The model contained **2,572,280 parameters**.
- The detection head used three scales with strides `8`, `16`, and `32`.
- The main output tensor was `[1, 84, 8400]`.
- The inspected criterion was `E2ELoss`, combining one-to-many and one-to-one loss branches with weights that change during training.
- The printed training hyperparameters were Ultralytics defaults and were not verified as the checkpoint's original training settings.
- The inspection examined model structure and configuration without conducting a training session.

Overall, the investigation established a practical understanding of the model's architecture, tensor flow, parameter values, prediction head, and high-level loss organization.

### Recommended Further Investigation

A deeper follow-up could focus on:

1. **Loss calculations:** Examine the complete `v8DetectionLoss` implementation, including box regression, classification, and task-aligned assignment.
2. **Inference:** Run the model on real images and inspect the decoded bounding boxes, confidence scores, and class predictions.
3. **Training:** Conduct a controlled training experiment to observe how loss values and model parameters change over time.