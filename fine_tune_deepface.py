import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import os
import pathlib
import random
from tqdm import tqdm


def build_model(num_classes: int, depth: str = 'full') -> Model:
    """Builds a VGG16-based model with a classification head."""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    if depth == 'medium':
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output, name='vgg16_medium')
    elif depth == 'shallow':
        base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output, name='vgg16_shallow')

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', name="fc1")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax', name="predictions")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def load_and_preprocess_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Loads, decodes, resizes, and normalizes an image."""
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
    except tf.errors.InvalidArgumentError:
        tf.print("Warning: Could not decode image, skipping:", path)
        image = tf.zeros([224, 224, 3], dtype=tf.float32)

    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_ytf_dataset(
    data_dir: str,
    batch_size: int,
    max_subjects: int | None = None,
    max_frames_per_video: int | None = None
) -> tuple[tf.data.Dataset, int]:
    """
    Creates a tf.data.Dataset for the YouTube Faces dataset, with options
    to limit the number of subjects and frames per video.

    Args:
        data_dir: Path to the root of the aligned_images_DB directory.
        batch_size: The batch size for the dataset.
        max_subjects: Optional[int]. If set, randomly selects this many subjects.
        max_frames_per_video: Optional[int]. If set, samples up to this many
                              frames per video directory (sorted).

    Returns:
        A tuple containing:
            - The configured tf.data.Dataset.
            - The number of classes (subjects) actually used.
    """
    data_root = pathlib.Path(data_dir)
    if not data_root.is_dir():
        raise ValueError(f"Data directory not found: {data_dir}")

    all_subject_paths = sorted([item for item in data_root.glob('*/') if item.is_dir()])
    if not all_subject_paths:
        raise ValueError(f"No subject directories found in {data_dir}.")

    if max_subjects and max_subjects > 0 and max_subjects < len(all_subject_paths):
        print(f"Limiting to a random sample of {max_subjects} subjects (out of {len(all_subject_paths)}).")
        selected_subject_paths = random.sample(all_subject_paths, max_subjects)
    else:
        print(f"Using all {len(all_subject_paths)} subjects.")
        selected_subject_paths = all_subject_paths

    num_classes = len(selected_subject_paths)
    if num_classes == 0:
         raise ValueError(f"No subjects selected or found.")

    print(f"Selected {num_classes} classes (subjects).")

    subject_to_index = {path.name: i for i, path in enumerate(selected_subject_paths)}

    all_image_paths = []
    all_image_labels = []
    print("Collecting image paths...")

    for subject_path in tqdm(selected_subject_paths, desc="Subjects"):
        subject_name = subject_path.name
        subject_label = subject_to_index[subject_name]
        video_paths = sorted([v for v in subject_path.glob('*/') if v.is_dir()])

        for video_path in video_paths:
            frame_paths_full = sorted([str(p) for p in video_path.glob('*.jpg')])

            if max_frames_per_video and max_frames_per_video > 0 and len(frame_paths_full) > max_frames_per_video:
                 selected_frame_paths = frame_paths_full[:max_frames_per_video]
            else:
                selected_frame_paths = frame_paths_full

            all_image_paths.extend(selected_frame_paths)
            all_image_labels.extend([subject_label] * len(selected_frame_paths))


    if not all_image_paths:
          raise ValueError(f"No images found for the selected subjects/frame limits within {data_dir}")

    print(f"Collected {len(all_image_paths)} images for training.")

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

    image_label_ds = tf.data.Dataset.zip((path_ds, label_ds))

    dataset = image_label_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=max(1000, len(all_image_paths) // 10))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset, num_classes

def train_model(
    data_dir: str,
    depth: str,
    epochs: int = 10,
    batch_size: int = 16,
    max_subjects: int | None = None,
    max_frames_per_video: int | None = None
):
    """Loads YTF data, builds the model, and trains it."""

    print("\n--- Creating Dataset ---")
    dataset, num_classes = create_ytf_dataset(
        data_dir,
        batch_size,
        max_subjects=max_subjects,
        max_frames_per_video=max_frames_per_video
    )
    print("--- Dataset Created ---")


    model = build_model(num_classes, depth=depth)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"\nStarting training for {epochs} epochs...")

    history = model.fit(
        dataset,
        epochs=epochs
    )
    print("Training finished.")

    return model, history

if __name__ == "__main__":
    ytf_aligned_data_dir = './YTF/aligned_images_DB'

    model_depth = 'medium'

    num_epochs = 15
    batch_size = 16

    MAX_SUBJECTS_TO_USE = 150
    MAX_FRAMES_PER_VIDEO = 20

    model_save_dir = './YTF/models'
    os.makedirs(model_save_dir, exist_ok=True)

    subset_info = f"subj{MAX_SUBJECTS_TO_USE or 'All'}_frames{MAX_FRAMES_PER_VIDEO or 'All'}"
    model_save_path = os.path.join(model_save_dir, f'ytf_vgg16_{model_depth}_{subset_info}_epochs{num_epochs}.h5')

    if not os.path.isdir(ytf_aligned_data_dir):
        print(f"ERROR: Data directory not found at '{ytf_aligned_data_dir}'")
    else:
        print(f"Using YTF aligned data directory: {ytf_aligned_data_dir}")
        print(f"Training model with depth: {model_depth}")
        print(f"Dataset limits: Subjects={MAX_SUBJECTS_TO_USE}, Frames/Video={MAX_FRAMES_PER_VIDEO}")
        print(f"Batch size: {batch_size}, Epochs: {num_epochs}")

        trained_model, training_history = train_model(
            data_dir=ytf_aligned_data_dir,
            depth=model_depth,
            epochs=num_epochs,
            batch_size=batch_size,
            max_subjects=MAX_SUBJECTS_TO_USE,
            max_frames_per_video=MAX_FRAMES_PER_VIDEO
        )

        print(f"Saving model to: {model_save_path}")
        trained_model.save(model_save_path)
        print("Model saved successfully.")

        final_train_accuracy = training_history.history['accuracy'][-1]
        print(f"\nFinal training accuracy: {final_train_accuracy*100:.2f}%")