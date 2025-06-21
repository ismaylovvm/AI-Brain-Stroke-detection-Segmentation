import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, Callback, ReduceLROnPlateau, EarlyStopping
from google.colab import drive
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import pandas as pd

drive.mount('/content/drive')

# ZIP file
zip_path = '/content/drive/My Drive/brain_stroke.zip'
unzip_path = '/content/brain_stroke'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_path)

# data path
image_dir = "/content/brain_stroke/brain_stroke/image"
mask_dir = "/content/brain_stroke/brain_stroke/maske"


target_size = (256, 256)


def load_data(image_dir, mask_dir, target_size):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    images = []
    masks = []

    for img_file, mask_file in zip(image_files, mask_files):
        img = cv2.imread(os.path.join(image_dir, img_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Uyarı: {img_file} veya {mask_file} dosyası okunamadı, atlanıyor.")
            continue

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        images.append(img / 255.0)
        masks.append(mask)

    return np.array(images).reshape(-1, target_size[0], target_size[1], 1), np.array(masks).reshape(-1, target_size[0], target_size[1], 1)

#  U-Net 
def improved_unet_model(input_size=(256, 256, 1), dropout_rate=0.2):
    inputs = Input(input_size)

    c1 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2,2))(c1)
    p1 = Dropout(dropout_rate)(p1)

    c2 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2,2))(c2)
    p2 = Dropout(dropout_rate)(p2)

    c3 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2,2))(c3)
    p3 = Dropout(dropout_rate)(p3)

    c4 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = Dropout(dropout_rate)(c4)

    u5 = UpSampling2D((2,2))(p4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(dropout_rate)(c5)

    u6 = UpSampling2D((2,2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(dropout_rate)(c6)

    u7 = UpSampling2D((2,2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    outputs = Conv2D(1, (1,1), activation='sigmoid')(c7)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=bce_dice_loss, metrics=[dice_coef])
    return model

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

class F1ScoreCallback(Callback):
    def _init_(self, validation_data):
        super(F1ScoreCallback, self)._init_()
        self.X_val, self.Y_val = validation_data
        self.f1_scores = []
        self.precision_scores = []
        self.recall_scores = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, batch_size=16, verbose=0)
        y_pred = (y_pred > 0.5).astype(np.uint8)
        y_true = self.Y_val.astype(np.uint8)

        
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        f1 = f1_score(y_true_flat, y_pred_flat)
        precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)

        self.f1_scores.append(f1)
        self.precision_scores.append(precision)
        self.recall_scores.append(recall)

        print(f"\nEpoch {epoch+1}: val_f1_score = {f1:.4f}, precision = {precision:.4f}, recall = {recall:.4f}")

  
def on_train_end(self, logs=None):
        
        best_f1_index = np.argmax(self.f1_scores)
        best_f1 = self.f1_scores[best_f1_index]
        best_epoch = best_f1_index + 1

        print(f"\nEn iyi F1 skoru: {best_f1:.4f} (Epoch {best_epoch})")

# 
def create_data_generators(X_train, Y_train, batch_size=8):
    seed = 42

    
    image_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    
    mask_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    
    image_generator = image_datagen.flow(
        X_train, seed=seed, batch_size=batch_size, shuffle=True
    )

    mask_generator = mask_datagen.flow(
        Y_train, seed=seed, batch_size=batch_size, shuffle=True
    )

    
    return zip(image_generator, mask_generator)


def clear_memory():
    import gc
    gc.collect()
    tf.keras.backend.clear_session()


print("Verileri yükleniyor...")
X, Y = load_data(image_dir, mask_dir, target_size)
print(f"Yüklenen veri boyutu: {X.shape}, {Y.shape}")


print("Veri doğruluğu kontrol ediliyor...")
if np.isnan(X).any() or np.isnan(Y).any():
    print("Uyarı: Verilerde NaN değerler var! İşleniyor...")
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)


from sklearn.model_selection import train_test_split

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

print(f"Eğitim seti: {X_train.shape}, Doğrulama seti: {X_val.shape}, Test seti: {X_test.shape}")


del X, Y, X_temp, Y_temp
clear_memory()


BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 5  # Early stopping 


model = improved_unet_model(dropout_rate=0.3)
model.summary()


checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
    mode='min'
)

csv_logger = CSVLogger('training_log.csv')

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6,
    mode='min'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    verbose=1,
    restore_best_weights=True,
    mode='min'
)

f1_callback = F1ScoreCallback(validation_data=(X_val, Y_val))

# Customize a generator that returns numpy arrays instead of a zip object
def generator_wrapper(generator, steps_per_epoch):
    while True:
        X_batch, Y_batch = next(generator)
        yield X_batch, Y_batch

# Create generator
train_generator_raw = create_data_generators(X_train, Y_train, batch_size=BATCH_SIZE)
steps_per_epoch = len(X_train) // BATCH_SIZE
train_generator = generator_wrapper(train_generator_raw, steps_per_epoch)


history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=(X_val, Y_val),
    callbacks=[checkpoint, csv_logger, f1_callback, reduce_lr, early_stopping],
    verbose=1
)

 
model.load_weights('best_model.h5')


print("Test seti üzerinde model değerlendiriliyor...")
test_loss, test_dice = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Dice Coefficient: {test_dice:.4f}")

# Test F1 Score
y_pred_test = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred_test_binary = (y_pred_test > 0.5).astype(np.uint8)
test_f1 = f1_score(Y_test.flatten(), y_pred_test_binary.flatten())
test_precision = precision_score(Y_test.flatten(), y_pred_test_binary.flatten(), zero_division=0)
test_recall = recall_score(Y_test.flatten(), y_pred_test_binary.flatten(), zero_division=0)

print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Loss & Dice Coefficient
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['dice_coef'], label='Train Dice')
plt.plot(history.history['val_dice_coef'], label='Val Dice')
plt.legend()
plt.title('Dice Coefficient')

plt.savefig('training_metrics.png')
plt.show()

# F1, Precision ve Recall Grafiği
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(f1_callback.f1_scores, label='Val F1 Score')
plt.legend()
plt.title('Validation F1 Score')

plt.subplot(1, 3, 2)
plt.plot(f1_callback.precision_scores, label='Val Precision')
plt.legend()
plt.title('Validation Precision')

plt.subplot(1, 3, 3)
plt.plot(f1_callback.recall_scores, label='Val Recall')
plt.legend()
plt.title('Validation Recall')

plt.savefig('f1_precision_recall.png')
plt.show()

# Confusion Matrix 
cm = confusion_matrix(Y_test.flatten(), y_pred_test_binary.flatten())
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(figsize=(10, 8))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.savefig('confusion_matrix.png')
plt.show()


def plot_sample_results(X_test, Y_test, y_pred, num_samples=3):
    indices = np.random.choice(range(len(X_test)), num_samples, replace=False)

    plt.figure(figsize=(15, 5 * num_samples))

    for i, idx in enumerate(indices):
        # Orijinal görüntü
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(X_test[idx].reshape(target_size), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Gerçek maske
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(Y_test[idx].reshape(target_size), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        # Tahmin edilen maske
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(y_pred[idx].reshape(target_size), cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

    plt.savefig('sample_predictions.png')
    plt.show()


plot_sample_results(X_test, Y_test, y_pred_test_binary, num_samples=3)


import pandas as pd

metrics_df = pd.DataFrame({
    'epoch': range(1, len(f1_callback.f1_scores) + 1),
    'f1_score': f1_callback.f1_scores,
    'precision': f1_callback.precision_scores,
    'recall': f1_callback.recall_scores
})

metrics_df.to_csv('validation_metrics.csv', index=False)

print("Eğitim tamamlandı! Sonuçlar kaydedildi.")
