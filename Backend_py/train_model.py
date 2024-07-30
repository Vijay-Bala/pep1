import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

faces_directory = 'static/faces'

dataset = image_dataset_from_directory(
    faces_directory,
    labels='inferred',
    label_mode='int',
    image_size=(256, 256),
    batch_size=32,
    shuffle=True,
    seed=123
)

val_split = 0.2
dataset_size = tf.data.experimental.cardinality(dataset).numpy()
train_size = int((1 - val_split) * dataset_size)
val_size = dataset_size - train_size
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize_with_pad(image, 224, 224)  
    return image, label

train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def create_model(num_classes):
    base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


num_classes = len(dataset.class_names)

model = create_model(num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

#change
epochs = 5
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

model_save_path = 'static/face_detection_model.keras'
model.save(model_save_path)

print(f"Model saved to {model_save_path}")
