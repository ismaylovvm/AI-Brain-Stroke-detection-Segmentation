# EfficientNetB0 Training Report

## 1. Dataset Preparation

- **DICOM images** with and without stroke findings were organized into **two separate folders**.
- Images were read using the **`pydicom`** library and converted to **Hounsfield Units (HU)**.
- All images were resized to **224×224 pixels** to match the input size requirements of the model architecture.
- Labeling:
  - Images with stroke findings were labeled as **1**.
  - Images without stroke findings were labeled as **0**.
- Corresponding arrays for images and labels were created.
- The dataset was split into **80% training and 20% testing** using **stratified sampling** to preserve class balance.

## 2. Data Augmentation and Preprocessing

- To reduce overfitting and increase data diversity, several augmentation techniques were applied to the training set:
  - **10% zoom**
  - **Horizontal and vertical flipping**
  - **10% horizontal and vertical translation**
  - **10% contrast adjustment**
  - **Random rotation**
- All pixel intensity values were **normalized to the [0, 1] range**.

## 3. Model Architecture

- The model was based on **EfficientNetB0** from **Keras applications**.
- The classifier head included:
  - **Global Average Pooling**
  - **Batch Normalization**
  - **Dropout** with a rate of **0.3**
- The final output layer had:
  - **1 neuron** with **sigmoid activation** for **binary classification**

## 4. Training Configuration

- **Initial training** was performed with:
  - **Batch size**: 32
  - **Validation split**: 20%
  - **Loss function**: Binary Cross-Entropy
  - **Optimizer**: Adam
  - **Learning rate**: 1×10⁻³
  - **Metrics**: Accuracy and AUC

- **Fine-tuning phase**:
  - The last **50 layers** of EfficientNetB0 were **unfrozen**.
  - Learning rate was reduced to **1×10⁻⁴**
  - Model retraining was conducted to improve performance.

## 5. Evaluation

- The model was evaluated on the **test dataset**.
- Metrics used for assessment:
  - **Accuracy**
  - **AUC**
  - **F1 Score**
  - **ROC Curve Analysis**
- An optimal classification **threshold of 0.5** was used during final evaluation.
