import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

train_dir = './VGG-Face2/data/test'

# Load datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=16,
    label_mode='int'
)

class_names = train_dataset.class_names
num_classes = len(class_names)

# Normalize imagesa
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and prefetch
train_dataset = train_dataset.shuffle(1000).prefetch(tf.data.AUTOTUNE)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

custom_model = Model(inputs=base_model.input, outputs=x)

custom_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

custom_model.summary()

custom_model.fit(train_dataset, epochs=20, batch_size=16)