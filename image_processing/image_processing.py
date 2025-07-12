import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_modality_lut

def read_dicom(file_path):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DICOM dosyası bulunamadı: {file_path}")
    dicom = pydicom.dcmread(file_path)
    hu_image = apply_modality_lut(dicom.pixel_array, dicom)
    return hu_image

def window_image(hu_image, window_center, window_width):
    
    lower_limit = window_center - (window_width / 2)
    upper_limit = window_center + (window_width / 2)
    windowed = np.clip(hu_image, lower_limit, upper_limit)
    normalized = (windowed - lower_limit) / (upper_limit - lower_limit)
    return normalized

def resize_image(image, size=(256, 256)):  
    
    return cv2.resize(image, size)

def convert_to_rgb(image):
    """
    0-1 aralığındaki gri görüntüyü 3 kanallı RGB görüntüye çevirir.
    """
    image_uint8 = (image * 255).astype(np.uint8)
    return cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

def apply_clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    Gri görüntüye CLAHE uygulayarak kontrastı iyileştirir.
    """
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_image = clahe.apply(image)
    normalized = clahe_image / 255.0
    return clahe_image, normalized

def plot_results(windowed_image, clahe_normalized):
    """
    Sonuç görüntülerini 1x2 subplot şeklinde görselleştirir.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].imshow(windowed_image, cmap='gray')
    axs[0].set_title("Pencerelenmiş Görüntü")
    axs[0].axis('off')

    axs[1].imshow(clahe_normalized, cmap='gray')
    axs[1].set_title("CLAHE Uygulanmış Görüntü")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    
    file_path = r"C:\Users\ismay\Downloads\Desktop\inme\Kanama Veri Seti\DICOM\10329.dcm"
    
    hu_image = read_dicom(file_path)
    
    window_center = 40
    window_width = 80  
    windowed_image = window_image(hu_image, window_center, window_width)
    
    resized_image = resize_image(windowed_image, (256, 256))  
    
    rgb_image = convert_to_rgb(resized_image)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    clahe_image, clahe_normalized = apply_clahe(gray_image, clipLimit=2.0, tileGridSize=(8, 8))
    
    plot_results(windowed_image, clahe_normalized)

if __name__ == "__main__":
    main()
