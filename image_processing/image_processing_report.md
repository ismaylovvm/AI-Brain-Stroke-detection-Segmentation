# Image Preprocessing Report

## 1. DICOM File Reading

- Medical DICOM files were read using the `pydicom` library.
- Pixel intensity values from these DICOM files were converted to **Hounsfield Units (HU)** using the `apply_modality_lut` method.
- This conversion ensured proper medical interpretation of the image data.

## 2. Windowing and Normalization

- The HU values were normalized through a **windowing process**.
- A specific intensity range was applied to enhance contrast in medically relevant regions.
- This windowing step helped emphasize stroke-related features in the images.

## 3. RGB Conversion

- The normalized images were converted into **3-channel RGB format**.
- This was necessary to comply with the **input requirements** of certain deep learning model architectures.

## 4. Contrast Enhancement with CLAHE

- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** was applied.
- This technique improves local contrast and highlights fine details in medical images.
- CLAHE helps reveal subtle abnormalities such as ischemic or hemorrhagic stroke patterns.

## 5. Resizing

- All images were resized to **256×256 pixels** using **OpenCV’s `resize` method**.
- This ensures compatibility with the model input layer and allows for efficient batch processing during training and inference.
