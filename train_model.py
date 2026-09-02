import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# ==========================================
# 1. إعدادات المشروع والمتغيرات الأساسية
# ==========================================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25
DATA_DIR = "dataset"  # مسار مجلد البيانات
MODEL_SAVE_PATH = "plant_disease_model.h5"

print("--- الخطوة 1 & 2: تحميل وتقسيم البيانات (Train / Validation / Test) ---")

# تحميل بيانات التدريب والتحقق (80% تدريب، 20% تحقق)
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
print(f"الفئات المكتشفة في الداتا سيت: {class_names}")

# تقسيم جزء من Validation ليكون Test حقيقي للتقييم النهائي
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_ds = val_ds.skip(val_batches // 2)

# ==========================================
# 3. المعالجة وتعزيز البيانات (Preprocessing & Data Augmentation)
# ==========================================
print("--- الخطوة 3 & 4: المعالجة وتعزيز البيانات ---")

normalization_layer = layers.Rescaling(1./255)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

train_ds = train_ds.map(lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 5. بناء هيكل الـ CNN Model
# ==========================================
print("--- الخطوة 5: بناء نموذج الـ CNN ---")

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # Classifier Head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 6. إعداد الـ Callbacks لحفظ أفضل نموذج
# ==========================================
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

callbacks_list = [checkpoint, early_stopping]

# ==========================================
# 7 & 8. التدريب والتحقق (Training & Validation)
# ==========================================
print("--- الخطوة 6 & 7: بدء التدريب والتحقق الفعلي ---")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_list
)

# ==========================================
# 9. التقييم النهائي (Evaluation على الـ Test Set)
# ==========================================
print("--- الخطوة 8: تقييم النموذج على مجموعة الاختبار (Test Evaluation) ---")

best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

test_loss, test_accuracy = best_model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print(f"--- الخطوة 10: تم حفظ أفضل نموذج بنجاح في المسار: {MODEL_SAVE_PATH} ---")
