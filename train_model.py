import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Basic Variables
# ==========================================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25
DATA_DIR = "dataset"

print("--- Step 1 & 2: Loading and Splitting Dataset (Train / Validation / Test) ---")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Detected Classes: {class_names}")

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_ds = val_ds.skip(val_batches // 2)

# ==========================================
# 3. Preprocessing & Comprehensive Data Augmentation
# ==========================================
print("--- Step 3 & 4: Preprocessing & Comprehensive Data Augmentation ---")

normalization_layer = layers.Rescaling(1./255)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
    layers.RandomBrightness(factor=0.2),
])

train_ds = train_ds.map(lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 4. Advanced Callbacks Configuration
# ==========================================
def get_callbacks(model_path):
    # حفظ أفضل نموذج بناءً على الـ validation accuracy
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    # إيقاف التدريب عند ثبات الـ validation loss لمنع الـ Overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    # تقليل معدل التعلم (Learning Rate) عندما يتوقف التحسن
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    return [checkpoint, early_stopping, reduce_lr]

# ==========================================
# 5. Model 1: Custom CNN
# ==========================================
print("\n=== Training Model 1: Custom CNN ===")
custom_cnn = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

custom_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_cnn = custom_cnn.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS, 
    callbacks=get_callbacks("custom_cnn_model.h5")
)

# ==========================================
# 6. Model 2: MobileNetV2 (Transfer Learning)
# ==========================================
print("\n=== Training Model 2: MobileNetV2 ===")
base_mobilenet = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_mobilenet.trainable = False

inputs = layers.Input(shape=(128, 128, 3))
x = base_mobilenet(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

mobilenet_model = Model(inputs, outputs)
mobilenet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_mobilenet = mobilenet_model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS, 
    callbacks=get_callbacks("mobilenetv2_model.h5")
)

# ==========================================
# 7. Model 3: EfficientNetB0 (Transfer Learning)
# ==========================================
print("\n=== Training Model 3: EfficientNetB0 ===")
base_efficientnet = EfficientNetB0(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_efficientnet.trainable = False

inputs = layers.Input(shape=(128, 128, 3))
x = base_efficientnet(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

efficientnet_model = Model(inputs, outputs)
efficientnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_efficientnet = efficientnet_model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS, 
    callbacks=get_callbacks("efficientnet_model.h5")
)

print("\n--- All models trained and saved successfully! ---")

# ==========================================
# 8. Final Evaluation & Comparison
# ==========================================
models_dict = {
    "Custom CNN": "custom_cnn_model.h5",
    "MobileNetV2": "mobilenetv2_model.h5",
    "EfficientNetB0": "efficientnet_model.h5"
}

print("\n==================== COMPARISON RESULTS ====================")
for name, path in models_dict.items():
    if os.path.exists(path):
        loaded_model = tf.keras.models.load_model(path)
        loss, acc = loaded_model.evaluate(test_ds, verbose=0)
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Model: {name} | Test Accuracy: {acc*100:.2f}% | Test Loss: {loss:.4f} | Size: {file_size_mb:.2f} MB")
print("============================================================")

# ==========================================
# 9. Plot & Save Training Curves
# ==========================================
acc = history_cnn.history['accuracy']
val_acc = history_cnn.history['val_accuracy']
loss = history_cnn.history['loss']
val_loss = history_cnn.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Custom CNN - Training and Validation Accuracy')
plt.savefig('accuracy.png')
plt.close()

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Custom CNN - Training and Validation Loss')
plt.savefig('loss.png')
plt.close()

print("Training curves (accuracy.png & loss.png) saved successfully!")

