

import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from google.colab import drive
import numpy as np
import zipfile
import os
import shutil

# 1. Google Drive 
drive.mount('/content/drive')


ZIP_PATH = '/content/drive/MyDrive/ct_data/dataset.zip'
EXTRACT_PATH = '/content/drive/MyDrive/ct_data'

if os.path.exists(EXTRACT_PATH):
    shutil.rmtree(EXTRACT_PATH)
os.makedirs(EXTRACT_PATH)

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)


data_dir = os.path.join(EXTRACT_PATH, 'dataset')
all_images = []
all_labels = []

for class_name in ['0', '1']:
    class_dir = os.path.join(data_dir, class_name)
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
    all_images.extend(images)
    all_labels.extend([int(class_name)] * len(images))

X_train, X_test, y_train, y_test = train_test_split(
    all_images, all_labels,
    test_size=0.2,
    stratify=all_labels,
    random_state=42
)


TRAIN_DIR = '/content/train'
TEST_DIR = '/content/test'

def create_subset(subset_path, files, labels):
    if os.path.exists(subset_path):
        shutil.rmtree(subset_path)
    os.makedirs(subset_path)

    for class_name in ['0', '1']:
        os.makedirs(os.path.join(subset_path, class_name), exist_ok=True)

    for file_path, label in zip(files, labels):
        class_dir = os.path.join(subset_path, str(label))
        shutil.copy(file_path, class_dir)

create_subset(TRAIN_DIR, X_train, y_train)
create_subset(TEST_DIR, X_test, y_test)


IMG_SIZE = 224
BATCH_SIZE = 32

def preprocess_fn(img):
    return applications.efficientnet.preprocess_input(img)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_fn,
    validation_split=0.2  
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# (EfficientNetB0)
def build_model():
    base_model = applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)

model = build_model()


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

checkpoint = callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_auc',
    patience=10,
    restore_best_weights=True,
    verbose=1
)


history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)


model.save('base_model.keras')
print("\n✅ Temel model kaydedildi: base_model.keras")


model = models.load_model('best_model.keras')  


model.layers[1].trainable = True
for layer in model.layers[1].layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

history_fine = model.fit(
    train_generator,
    epochs=40,
    initial_epoch=history.epoch[-1],
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)


model.save('fine_tuned_model.keras')
print("\n✅ İnce ayarlı model kaydedildi: fine_tuned_model.keras")

best_model = models.load_model('best_model.keras')

test_results = best_model.evaluate(test_generator)
print(f"\n📊 Test Sonuçları: Accuracy: {test_results[1]:.4f}, AUC: {test_results[2]:.4f}")


best_model.save('final_model.keras')
best_model.save('final_model', save_format='tf')
print("\n🎉 Final model kaydedildi:")
print("- final_model.keras (Keras format)")
print("- final_model/ (TensorFlow SavedModel)")


y_pred = (best_model.predict(test_generator) > 0.5).astype(int)
print("\n📋 Sınıflandırma Raporu:")
print(classification_report(test_generator.labels, y_pred))
print(f"F1 Score: {f1_score(test_generator.labels, y_pred):.4f}")

