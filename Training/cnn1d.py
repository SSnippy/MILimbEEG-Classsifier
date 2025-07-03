import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout, GlobalAveragePooling1D, Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau

# variables to change
DOWNSAMPLE = 4  # Use every 4th row (adjust as needed)
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
NUM_FILTERS = 128  # Increased filters
KERNEL_SIZE = 5   # Slightly smaller kernel
DENSE_UNITS = 128 # Increased dense units
DROPOUT_RATE = 0.5

META_PATH = 'datapoints/metadata.xlsx'

# Load metadata
meta_df = pd.read_excel(META_PATH)
num_classes = meta_df['task_label_encoded'].nunique()

# Load all EEG data
X = []
y = []
for idx, row in meta_df.iterrows():
    eeg_path = row['local_url']
    if not os.path.exists(eeg_path):
        print(f"File not found: {eeg_path}")
        continue
    df = pd.read_csv(eeg_path)
    eeg_data = df.select_dtypes(include=[np.number]).values
    # Downsample
    eeg_data = eeg_data[::DOWNSAMPLE]
    # Add task_type as a feature
    task_type = np.full((eeg_data.shape[0], 1), row['task_type'])
    eeg_data = np.hstack([eeg_data, task_type])
    X.append(eeg_data)
    y.append(row['task_label_encoded'])

# Pad sequences to the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences
maxlen = max([x.shape[0] for x in X])
X_pad = pad_sequences(X, maxlen=maxlen, dtype='float32', padding='post', truncating='post')
y = np.array(y)

# Scale features
n_features = X_pad.shape[2]
scaler = StandardScaler()
X_reshaped = X_pad.reshape(-1, n_features)
X_scaled = scaler.fit_transform(X_reshaped).reshape(X_pad.shape)

# Split features
X_eeg = X_scaled[:, :, :-1]  # all but last column (EEG channels)
X_task = X_scaled[:, 0, -1].reshape(-1, 1)  # task_type, take from first time step

# Train/test split
X_eeg_train, X_eeg_test, X_task_train, X_task_test, y_train, y_test = train_test_split(
    X_eeg, X_task, y, test_size=0.2, stratify=y, random_state=42
)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Build 1D CNN model
input_eeg = Input(shape=(maxlen, n_features-1), name='eeg_input')
x = Conv1D(NUM_FILTERS, KERNEL_SIZE, padding='same')(input_eeg)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(DROPOUT_RATE)(x)
x = Conv1D(NUM_FILTERS, KERNEL_SIZE, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(DROPOUT_RATE)(x)
x = GlobalAveragePooling1D()(x)

input_task = Input(shape=(1,), name='task_type_input')
concat = Concatenate()([x, input_task])
dense = Dense(256, activation='relu')(concat)
dense = BatchNormalization()(dense)
dense = Dropout(DROPOUT_RATE)(dense)
dense = Dense(DENSE_UNITS, activation='relu')(dense)
dense = Dropout(DROPOUT_RATE)(dense)
output = Dense(num_classes, activation='softmax')(dense)

model = Model(inputs=[input_eeg, input_task], outputs=output)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)

history = model.fit(
    [X_eeg_train, X_task_train], y_train_cat,
    validation_data=([X_eeg_test, X_task_test], y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[lr_scheduler]
)

# Plot accuracy and loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.tight_layout()
plt.show()

# Final evaluation
loss, acc = model.evaluate([X_eeg_test, X_task_test], y_test_cat, verbose=0)
print(f"Final Test Accuracy: {acc:.4f}")

import joblib
joblib.dump(model,'cnn1d_model')
joblib.dump(history.history,'cnn1d_history') 