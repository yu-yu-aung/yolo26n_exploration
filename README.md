# YOLO26n Model Exploration 🔬

A research and study-only repository dedicated to exploring and understanding the internal architecture of the **YOLO26n object-detection model** using Python, PyTorch, and Ultralytics.

The goal of this project is to understand how YOLO works internally rather than treating it as a black box. Through hands-on experiments and source-code inspection, this repository explores the model's architecture, tensor flow, learned parameters, detection head, configuration, and loss function.

## 🎯 Project Objectives

This repository explores the following aspects of YOLO26n:

- **Model Architecture:** Understand the backbone, neck, detection head, and individual layer types.
- **Tensor Flow:** Trace layer connections and inspect input and output tensor shapes.
- **Model Parameters:** Examine learned weights, biases, BatchNorm parameters, and parameter counts.
- **Model Configuration:** Investigate model attributes, training hyperparameters, and data augmentation settings.
- **Detection Head:** Understand multi-scale feature maps, prediction tensors, and detection head components.
- **Loss Function:** Explore how YOLO initializes its training criterion and combines one-to-many and one-to-one loss branches.

## 📚 Exploration Roadmap

The exploration follows a step-by-step approach, starting with loading the model and gradually moving toward its internal components.

Begin with **`01_load_model`** and follow the numbered notebooks or scripts in sequence.

Each stage builds upon the previous one to develop a clearer understanding of how YOLO26n works internally.

## 🛠️ Technologies

- Python
- PyTorch
- Ultralytics
- YOLO26n

## ⚠️ Disclaimer

This repository is intended **strictly for research, learning, and educational purposes**.

It focuses on inspecting a pretrained YOLO26n checkpoint and understanding its internal implementation. It is not intended to provide a complete implementation of YOLO or to reproduce the model's original training process.

Some configuration values examined in this project come from the installed Ultralytics package and may not represent the exact settings used to train the original checkpoint.

## 💬 Feedback & Discussion

I'm a Computer Science student exploring computer vision and deep learning, and this repository is part of my learning journey.

If you're interested in YOLO, object detection, or deep learning internals, feel free to reach out! I'd be happy to discuss ideas, exchange knowledge, and learn from others.

**Constructive feedback, suggestions, corrections, and technical discussions are always welcome.** I'm still learning, so different perspectives and insights are greatly appreciated.

Thanks for stopping by, and happy exploring! 🚀