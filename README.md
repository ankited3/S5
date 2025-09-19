# Session 5

## 🧠 MNIST CNN with PyTorch

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** for digit classification on the **MNIST dataset**.  
The model is lightweight yet effective, with **<20K parameters**, and achieves **>99.4% test accuracy** in **<20  epochs**.


---

# 📌 Iteration 1

The network consists of **7 convolutional layers** followed by **1 fully connected layer**.  
Non-linearities are introduced using **ReLU** activations, and **max pooling** is applied at two points for spatial downsampling.  
The final classification layer outputs **log-probabilities** using `log_softmax`.

---

## 🏗️ Architecture

### Convolutional Layers
1. **Conv1:** `1 → 16` channels, `3x3` kernel, padding=1  
2. **Conv2:** `16 → 16` channels, `3x3` kernel, padding=1  
3. **Conv3:** `16 → 32` channels, `3x3` kernel, padding=1  
   - **Max Pooling (2x2)**  
4. **Conv4:** `32 → 16` channels, `1x1` kernel  
5. **Conv5:** `16 → 32` channels, `3x3` kernel, padding=1  
   - **Max Pooling (2x2)**  
6. **Conv6:** `32 → 16` channels, `1x1` kernel  
7. **Conv7:** `16 → 16` channels, `3x3` kernel  

---

### Fully Connected Layer
- **FC1:** Input: `400` → Output: `10` (number of classes)

- # 📊 Layer-by-Layer Summary (with Receptive Field)

The table below shows **how the receptive field grows** across layers along with the main parameters:

| Layer    | Input Ch. | Output Ch. | Kernel | Stride | Padding | Output Size | Receptive Field |
|----------|-----------|------------|--------|--------|---------|-------------|-----------------|
| Input    | 1         | 1          | -      | -      | -       | 28×28       | 1×1             |
| Conv1    | 1         | 16         | 3×3    | 1      | 1       | 28×28       | 3×3             |
| Conv2    | 16        | 16         | 3×3    | 1      | 1       | 28×28       | 5×5             |
| Conv3    | 16        | 32         | 3×3    | 1      | 1       | 28×28       | 7×7             |
| MaxPool1 | 32        | 32         | 2×2    | 2      | 0       | 14×14       | 8×8             |
| Conv4    | 32        | 16         | 1×1    | 1      | 0       | 14×14       | 8×8             |
| Conv5    | 16        | 32         | 3×3    | 1      | 1       | 14×14       | 12×12           |
| MaxPool2 | 32        | 32         | 2×2    | 2      | 0       | 7×7         | 14×14           |
| Conv6    | 32        | 16         | 1×1    | 1      | 0       | 7×7         | 14×14           |
| Conv7    | 16        | 16         | 3×3    | 1      | 0       | 5×5         | 18×18           |
| Flatten  | 16×5×5    | -          | -      | -      | -       | 400         | -               |
| FC1      | 400       | 10         | -      | -      | -       | 10          | Global          |

### Number of parameters:19146

✅ This table helps visualize how **local pixel coverage expands** until the FC layer aggregates global features.  


---



Applied using `torchvision.transforms`:

1. **RandomApply([CenterCrop(22)], p=0.1)**  
   - With probability 10%, randomly crops the image to 22×22 before resizing back.  
   - Introduces robustness to partial digits.  

2. **Resize((28, 28))**  
   - Ensures the image is always resized back to 28×28.  

3. **RandomRotation((-10°, +10°), fill=0)**  
   - Random rotation within ±10 degrees.  
   - `fill=0` ensures background stays black.  

4. **Normalize((0.1307,), (0.3081,))**  
   - Standard MNIST normalization (mean=0.1307, std=0.3081).  

---

## 🧪 Test Data Preprocessing

For evaluation, we only apply:


1. **Normalize((0.1307,), (0.3081,))**  
   - Standard MNIST normalization (mean=0.1307, std=0.3081).   

No augmentations are used during testing.

## ⚡ Training Setup

 **Adam optimizer** with cross-entropy loss for training.

---



## 📊 Training & Test Results

The model was trained on **MNIST** for **19 epochs** using the defined architecture, data augmentations, and training setup.

---

## 📈 Accuracy Summary

- **Train Accuracy:**  
  - Approached **98.59%** after 19 epochs  

- **Test Accuracy:**  
  - Reached **99.16%** after ~19 epochs  


---

## 🔍 Observations

- The model generalizes very well, with **test accuracy slightly higher than training accuracy** (likely due to augmentations providing better regularization).  
- The **plateau** around 99.15% test accuracy indicates that the network has nearly saturated MNIST performance.  
- Further improvements may require:  
  - Additional **regularization** (e.g., Dropout, batch normalization)  
  - More **aggressive data augmentation**  
  -  **learning rate schedules** (e.g. stepLR, OneCycleLR, Cosine Annealing)
 

# 📌 Iteration 2
  ## 🧠 Updated Neural Network with BatchNorm & Dropout

  ### Number of parameters:19418

This version of the model builds on the previous architecture with the addition of:

- **Batch Normalization** after each convolutional layer  
- **Dropout** before the fully connected layer  

These changes improve **training stability**, **convergence speed**, and **generalization**.

---

## 🏗️ Architecture

### Convolutional + BatchNorm Layers
1. **Conv1 + BN1:** `1 → 16`, `3×3`, padding=1  
2. **Conv2 + BN2:** `16 → 16`, `3×3`, padding=1  
3. **Conv3 + BN3:** `16 → 32`, `3×3`, padding=1  
   - **Max Pooling (2×2)**  
4. **Conv4 + BN4:** `32 → 16`, `1×1`  
5. **Conv5 + BN5:** `16 → 32`, `3×3`, padding=1  
   - **Max Pooling (2×2)**  
6. **Conv6 + BN6:** `32 → 16`, `1×1`  
7. **Conv7 + BN7:** `16 → 16`, `3×3`  

---

### Fully Connected + Dropout
- **Dropout1:** 10% (p=0.1)  
- **FC1:** Linear `400 → 10`  

---
# 📊 Layer-by-Layer Summary (with Receptive Field, BatchNorm & Dropout)

| Layer        | Input Ch. | Output Ch. | Kernel | Stride | Padding | Output Size | Receptive Field |
|--------------|-----------|------------|--------|--------|---------|-------------|-----------------|
| Input        | 1         | 1          | -      | -      | -       | 28×28       | 1×1             |
| Conv1 + BN1  | 1         | 16         | 3×3    | 1      | 1       | 28×28       | 3×3             |
| Conv2 + BN2  | 16        | 16         | 3×3    | 1      | 1       | 28×28       | 5×5             |
| Conv3 + BN3  | 16        | 32         | 3×3    | 1      | 1       | 28×28       | 7×7             |
| MaxPool1     | 32        | 32         | 2×2    | 2      | 0       | 14×14       | 8×8             |
| Conv4 + BN4  | 32        | 16         | 1×1    | 1      | 0       | 14×14       | 8×8             |
| Conv5 + BN5  | 16        | 32         | 3×3    | 1      | 1       | 14×14       | 12×12           |
| MaxPool2     | 32        | 32         | 2×2    | 2      | 0       | 7×7         | 14×14           |
| Conv6 + BN6  | 32        | 16         | 1×1    | 1      | 0       | 7×7         | 14×14           |
| Conv7 + BN7  | 16        | 16         | 3×3    | 1      | 0       | 5×5         | 18×18           |
| Flatten      | 16×5×5    | -          | -      | -      | -       | 400         | -               |
| Dropout (p=0.1) | 400    | 400        | -      | -      | -       | 400         | -               |
| FC1          | 400       | 10         | -      | -      | -       | 10          | Global          |

### Number of parameters:19300


✅ This table now includes **BatchNorm** and **Dropout** alongside receptive field growth.

# 📊 Training & Test Results

---

## 📈 Accuracy Summary

- **Train Accuracy:**  
  - Approached **98.92%** after 19 epochs 

- **Test Accuracy:**  
  - Reached **99%** after ~19 epochs  
  - Plateaued around **99.3%**  



# 📌 Iteration 3

  ## 🧠 Added learning rate scheduler to 
  - Scheduler Type: StepLR

      Parameters:

     step_size=10 → LR decays every 10 epochs

     gamma=0.5 → LR is multiplied with 0.5 after each step

# 📊 Training & Test Results

---

## 📈 Accuracy Summary

- **Train Accuracy:**  
  - Approached **99.02%** after 19 epochs

- **Test Accuracy:**  
  - Reached **99.4%** after ~19 epochs  


 
 ## 🔍 Observations after iteration 3

- Test accuracy is dwelling in between 99% to 99.4% irrespecive of epochs numbers i.e. ideally accuracy should have increased with increase in epoch number.

 
# 📌 Iteration 4

  ## updated learning rate scheduler to 
  - Scheduler Type: StepLR

      Parameters:

     step_size=5 → LR decays every 5 epochs

     gamma=0.5 → LR is halved each step

# 📊 Training & Test Results

---

## 📈 Accuracy Summary

- **Train Accuracy:**  
  - Reached **99.1%** after ~19 epochs   

- **Test Accuracy:**  
  - Approached **99.4%** after first 6 epochs then 
 
  ## 🔍 Observations after iteration 4

-Test accuracy keeps increasing with epochs numbers

Epoch 1
Train: Loss=0.1812 Batch_id=468 Accuracy=92.21: 100%|██████████| 469/469 [00:30<00:00, 15.59it/s]

Test set: Average loss: 0.0007, Accuracy: 9722/10000 (97.22%)

Epoch 2
Train: Loss=0.0452 Batch_id=468 Accuracy=97.38: 100%|██████████| 469/469 [00:29<00:00, 15.72it/s]

Test set: Average loss: 0.0003, Accuracy: 9868/10000 (98.68%)

Epoch 3
Train: Loss=0.0539 Batch_id=468 Accuracy=97.83: 100%|██████████| 469/469 [00:29<00:00, 15.64it/s]

Test set: Average loss: 0.0004, Accuracy: 9831/10000 (98.31%)

Epoch 4
Train: Loss=0.0574 Batch_id=468 Accuracy=98.12: 100%|██████████| 469/469 [00:29<00:00, 15.73it/s]

Test set: Average loss: 0.0002, Accuracy: 9889/10000 (98.89%)

Epoch 5
Train: Loss=0.0455 Batch_id=468 Accuracy=98.26: 100%|██████████| 469/469 [00:30<00:00, 15.45it/s]

Test set: Average loss: 0.0002, Accuracy: 9915/10000 (99.15%)

Epoch 6
Train: Loss=0.0065 Batch_id=468 Accuracy=98.67: 100%|██████████| 469/469 [00:29<00:00, 15.78it/s]

Test set: Average loss: 0.0001, Accuracy: 9941/10000 (99.41%)
