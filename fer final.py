#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from IPython.display import display, Image
import numpy as np
from keras.models import Sequential, load_model  
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import pandas as pd  
def load_dataset(dataset_path):
    image_files = []
    labels = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                if os.path.isfile(image_path):
                    image_files.append(image_path)
                    labels.append(label)

    return image_files, labels
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_labels = {
    0: "Angry",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Sadness",
    6: "Surprise"
}
def display_cv2_image(image):
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()
    display(Image(data=img_bytes))
def recognize_emotion(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)
        emotion_probs = model.predict(face_roi)
        emotion_label = np.argmax(emotion_probs)
        emotion_string = emotion_labels[emotion_label]

        cv2.putText(frame, emotion_string, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame
def capture_frame(model):
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = recognize_emotion(frame, model)
            display_cv2_image(frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
image_files, labels = load_dataset("C:/Users/sampa/Downloads/archive (4)")
df = pd.DataFrame({'file_path': image_files, 'label': labels})
df['label'] = df['label'].astype(str)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='file_path',
    y_col='label',
    target_size=(48, 48),
    batch_size=32,
    class_mode='sparse',
    color_mode='grayscale')

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='file_path',
    y_col='label',
    target_size=(48, 48),
    batch_size=32,
    class_mode='sparse',
    color_mode='grayscale')
input_shape = (48, 48, 1) 
num_classes = 7 

cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(256, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(num_classes, activation='softmax'))

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

cnn_model.fit(train_generator, epochs=20, callbacks=[early_stopping, model_checkpoint], validation_data=val_generator)

best_model = load_model('best_model.h5')

capture_frame(best_model)


# In[ ]:




