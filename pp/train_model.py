import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# --------------------------------------------------------
# 📁 Paths
# --------------------------------------------------------
train_dir = "dataset/train"
val_dir = "dataset/validation"

# --------------------------------------------------------
# 🧠 Auto-detect classes
# --------------------------------------------------------
CLASS_NAMES = sorted(os.listdir(train_dir))
print(f"📂 Found {len(CLASS_NAMES)} classes:")
print(CLASS_NAMES)

# --------------------------------------------------------
# 🌱 Data Generators (with augmentation for better accuracy)
# --------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# --------------------------------------------------------
# 🧩 Model Architecture (Simple CNN)
# --------------------------------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASS_NAMES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --------------------------------------------------------
# 💾 Callbacks
# --------------------------------------------------------
checkpoint = ModelCheckpoint(
    'plant_disease_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# --------------------------------------------------------
# 🚀 Train Model
# --------------------------------------------------------
history = model.fit(
    train_data,
    epochs=25,
    validation_data=val_data,
    callbacks=[checkpoint, early_stop]
)

print("\n✅ Training complete! Best model saved as plant_disease_model.h5")

# --------------------------------------------------------
# 📊 Save class names for Streamlit app (optional but useful)
# --------------------------------------------------------
with open("class_names.txt", "w") as f:
    for cls in CLASS_NAMES:
        f.write(cls + "\n")

print("✅ Class names saved to class_names.txt")
