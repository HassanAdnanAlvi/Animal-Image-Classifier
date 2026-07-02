"""
=============================================================
  Animal Image Classifier – Dog vs Cat
  CSC-325 | Artificial Intelligence | Assignment No. 3
  Bahria University, Karachi Campus
=============================================================

Tool Stack : Python 3.x, TensorFlow/Keras, NumPy, Matplotlib,
             scikit-learn, tkinter (desktop GUI)
Dataset    : Kaggle "Dogs vs Cats" dataset
             https://www.kaggle.com/datasets/salader/dogs-vs-cats
             (25,000 labelled images: 12,500 cats, 12,500 dogs)
Methodology: Transfer Learning using MobileNetV2 (pre-trained on
             ImageNet) with a custom classification head.
=============================================================
"""

# ──────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score, recall_score)

# ──────────────────────────────────────────────
# 2. Configuration
# ──────────────────────────────────────────────
IMG_SIZE   = (160, 160)   # MobileNetV2 min input is 96×96
BATCH_SIZE = 32
EPOCHS     = 20
CLASSES    = ['cat', 'dog']
DATA_DIR   = "dataset"   # Unzip Kaggle dataset here; expects train/ and test/ sub-folders

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────
# 3. Data Preparation & Augmentation
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# 2.5 Auto-Clean Corrupted Dataset Images
# ──────────────────────────────────────────────
def clean_dataset(directory):
    from PIL import Image
    print(f"Checking for corrupted images in: {directory}...")
    bad_files = 0
    for root_dir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root_dir, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify() # Verify it is a valid image
                except Exception:
                    print(f"Removing corrupted image: {file_path}")
                    os.remove(file_path)
                    bad_files += 1
    print(f"Scan complete. Removed {bad_files} broken files.\n")

# Run the cleaner on both folders before training starts
clean_dataset(os.path.join(DATA_DIR, "train"))
clean_dataset(os.path.join(DATA_DIR, "test"))
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,   # 80/20 split from training folder
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ──────────────────────────────────────────────
# 4. Model Architecture – Transfer Learning
# ──────────────────────────────────────────────
def build_model():
    """
    MobileNetV2 base (frozen) + custom Dense head.
    Freezing the base preserves ImageNet features;
    only the head is trained on the animal dataset.
    """
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False    # Phase 1: freeze feature extractor

    inputs  = tf.keras.Input(shape=IMG_SIZE + (3,))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(64,  activation='relu')(x)
    x       = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)   # binary output

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

model, base_model = build_model()
model.summary()

# ──────────────────────────────────────────────
# 5. Phase 1 Training – Head Only
# ──────────────────────────────────────────────
#callbacks = [
#    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
#    ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss', verbose=1)
#]
#
#print("\n[Phase 1] Training classification head …")
#history1 = model.fit(
#    train_generator,
#    epochs=10,
#    validation_data=val_generator,
#    callbacks=callbacks,
#    verbose=1
#)
#
# ──────────────────────────────────────────────
# 6. Phase 2 Fine-Tuning – Unfreeze Top Layers
# ──────────────────────────────────────────────
base_model.trainable = True
# Only unfreeze the last 30 layers to avoid catastrophic forgetting
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),  # much lower LR
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n[Phase 2] Fine-tuning top layers …")
history2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# ──────────────────────────────────────────────
# 7. Evaluation – Accuracy, Precision, Recall
# ──────────────────────────────────────────────
print("\n[Evaluation] Running on test set …")
y_pred_prob = model.predict(test_generator, verbose=1)
y_pred      = (y_pred_prob > 0.5).astype(int).flatten()
y_true      = test_generator.classes

acc       = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)

print(f"\n{'='*40}")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")
print(f"{'='*40}")
print("\nDetailed Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 1: Confusion matrix ---
im = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Label'); axes[0].set_ylabel('True Label')
axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
axes[0].set_xticklabels(CLASSES); axes[0].set_yticklabels(CLASSES)
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i,j]), ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14)
plt.colorbar(im, ax=axes[0])

# --- Plot 2: Training curves ---
h1_epochs = len(history1.history['accuracy'])
h2_epochs = len(history2.history['accuracy'])
all_acc    = history1.history['accuracy']  + history2.history['accuracy']
all_vacc   = history1.history['val_accuracy'] + history2.history['val_accuracy']
x_range    = range(1, h1_epochs + h2_epochs + 1)

axes[1].plot(x_range, all_acc,  label='Train Acc', linewidth=2)
axes[1].plot(x_range, all_vacc, label='Val Acc',   linewidth=2)
axes[1].axvline(h1_epochs, color='gray', linestyle='--', label='Fine-tune start')
axes[1].set_title('Training History', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches='tight')
print("\nResults chart saved → results.png")
plt.show()

# ──────────────────────────────────────────────
# 8. Save the Model
# ──────────────────────────────────────────────
model.save("animal_classifier.h5")
print("Model saved → animal_classifier.h5")

# ──────────────────────────────────────────────
# 9. Desktop Classifier GUI (tkinter)
# ──────────────────────────────────────────────
def launch_desktop_app():
    """
    Minimal desktop GUI: user picks an image file and the
    model predicts whether it is a Cat or Dog with confidence%.
    Run this function AFTER training (model must be saved).
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from PIL import Image, ImageTk
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image

    MODEL_PATH = "animal_classifier.h5"
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train the model first.")
        return

    clf = load_model(MODEL_PATH)

    def classify_image():
        file_path = filedialog.askopenfilename(
            title="Select Animal Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if not file_path:
            return

        # Pre-process
        img = keras_image.load_img(file_path, target_size=IMG_SIZE)
        arr = keras_image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        prob   = float(clf.predict(arr, verbose=0)[0][0])
        label  = "Dog" if prob > 0.5 else "Cat"
        conf   = prob if prob > 0.5 else 1 - prob

        # Show in GUI
        pil_img = Image.open(file_path).resize((280, 280))
        tk_img  = ImageTk.PhotoImage(pil_img)
        img_label.config(image=tk_img)
        img_label.image = tk_img
        result_var.set(f"Prediction: {label}  ({conf*100:.1f}% confidence)")
        result_label.config(fg="#1a7340" if label == "Dog" else "#b03a2e")

    # Build window
    root = tk.Tk()
    root.title("Animal Classifier – CSC-325")
    root.geometry("360x480")
    root.resizable(False, False)
    root.config(bg="#f0f4f8")

    tk.Label(root, text="🐾 Animal Classifier", font=("Arial", 18, "bold"),
             bg="#f0f4f8", fg="#2c3e50").pack(pady=12)

    img_label = tk.Label(root, bg="#dde3ea", width=280, height=280)
    img_label.pack(pady=4)

    tk.Button(root, text="📂  Select Image", command=classify_image,
              font=("Arial", 12), bg="#2980b9", fg="white",
              relief="flat", padx=16, pady=6, cursor="hand2").pack(pady=12)

    result_var   = tk.StringVar(value="No image selected")
    result_label = tk.Label(root, textvariable=result_var,
                            font=("Arial", 13, "bold"), bg="#f0f4f8", fg="#555")
    result_label.pack()

    root.mainloop()


# Uncomment the line below to launch the GUI after training:
launch_desktop_app()
