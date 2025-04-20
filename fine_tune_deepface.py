import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

def build_model(num_classes: int, depth: str = 'full') -> Model:
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Chop model depending on depth
    if depth == 'medium':
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    elif depth == 'shallow':
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', name="F7")(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

def train_model(data_dir: str, num_classes: int, depth: str, epochs: int = 15):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(224, 224),
        batch_size=16,
        label_mode='int'
    )
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    dataset = dataset.shuffle(1000).prefetch(tf.data.AUTOTUNE)

    model = build_model(num_classes, depth=depth)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(dataset, epochs=epochs)
    return model

# # Example usage (one of the experiments):
if __name__ == "__main__":
    original_dataset_dir = './VGG-Face2/data_aligned'

    num_classes = len(tf.keras.utils.image_dataset_from_directory(original_dataset_dir).class_names)
    model = train_model(original_dataset_dir, num_classes=num_classes, depth='shallow', epochs=15)
    # Save the model
    model.save('./VGG-Face2/models/shallow.h5')

    test_ds = tf.keras.utils.image_dataset_from_directory(
        './VGG-Face2/5pct',
        image_size=(224, 224),
        batch_size=16,
        label_mode='int'
    )
    test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)

    results = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy: {results[1]*100:.2f}%, Error: {100 - results[1]*100:.2f}%")