# 🧠 Stroke Segmentation Using U-Net on DICOM Images

This project implements a deep learning model using a U-Net architecture to segment hemorrhagic and ischemic stroke regions from DICOM medical images.

---

## 📁 Dataset Preparation

- **Data Source**:
  - Hemorrhagic and ischemic DICOM images
  - Corresponding mask images (binary, grayscale)
  - Stored in separate folders

- **Preprocessing**:
  - All DICOM images resized to `256×256` pixels
  - Masks resized accordingly to match dimensions
  - Images stored as NumPy arrays in separate `X` (input) and `y` (mask) sets

- **Splitting**:
  - 80% Training set
  - 20% Testing set

---

## 🧠 Model Architecture

- **Frameworks**:
  - [TensorFlow](https://www.tensorflow.org/)
  - [Keras](https://keras.io/)

- **Model**: U-Net

### 🔽 Encoder
- 4 convolutional blocks with:
  - Filter sizes: 64 → 128 → 256 → 512
  - `Conv2D` + `ReLU` + `MaxPooling2D`

### 🔁 Bottleneck
- `Conv2D` with 512 filters

### 🔼 Decoder
- `UpSampling2D`
- `Concatenate` with corresponding encoder layer
- `Conv2D` operations to refine

### 🔚 Output
- Final layer: `Conv2D(1)` + `Sigmoid`
- Output: Single-channel binary segmentation mask

---

## 🏷️ Mask & Label Configuration

- Masks contain **two classes**:
  - 0 → Background
  - 1 → Stroke region
- Pixel values scaled from `[0, 255]` to `[0, 1]`
- Optionally converted to categorical using `keras.utils.to_categorical()`

---

## 🧪 Training Details

- **Loss Function**:
  - `BinaryCrossentropy`
  - `Dice Loss` (combined)

- **Optimizer**:
  - `Adam`

- **Metrics**:
  - Loss
  - Dice Coefficient
  - F1 Score

- **Batch Size**: `8`  
- **Epochs**: `20`

---

## 🛠️ Callbacks Used

- `ModelCheckpoint`: Saves best model weights  
- `EarlyStopping`: Prevents overfitting  
- `ReduceLROnPlateau`: Lowers learning rate on plateau  
- `CSVLogger`: Logs training history to CSV  
- `F1ScoreCallback`: Custom callback to compute F1 per epoch

---

## 📦 Output

- Model predicts binary mask for stroke segmentation
- Post-processed output masks are thresholded using sigmoid output
- Final output dimensions: `256 x 256 x 1`

---

## 📈 Example Performance

> *Coming soon: Add charts or sample results from test predictions*

---

## 🧾 References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- TensorFlow / Keras documentation
