import os
import cv2
import gc
from PIL import Image
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm

INPUT_DIR = './VGG-Face2/data/test'
OUTPUT_DIR = './VGG-Face2/data_aligned'

def align_face(image_path):
    detector = MTCNN()  # reinitialize per image
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    if results:
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        face = img_rgb[y:y+h, x:x+w]
        if face.size == 0:
            return None
        face_resized = cv2.resize(face, (224, 224))
        return Image.fromarray(face_resized)
    return None

def align_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    class_dirs = sorted(os.listdir(input_dir))
    for class_name in class_dirs:
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)

        if not os.path.isdir(input_class_dir):
            continue

        if os.path.exists(output_class_dir) and len(os.listdir(output_class_dir)) > 0:
            print(f"Skipping {class_name}, already processed.")
            continue

        os.makedirs(output_class_dir, exist_ok=True)
        file_list = os.listdir(input_class_dir)

        for i, filename in enumerate(tqdm(file_list, desc=f'Processing {class_name}', ncols=100)):
            input_path = os.path.join(input_class_dir, filename)
            output_path = os.path.join(output_class_dir, filename)

            if os.path.exists(output_path):
                continue

            aligned_face = align_face(input_path)
            if aligned_face:
                with open(output_path, 'wb') as f:
                    aligned_face.save(f, format='JPEG')

            # Garbage collection every 50 images
            if i % 50 == 0:
                gc.collect()
                cv2.destroyAllWindows()

align_dataset(INPUT_DIR, OUTPUT_DIR)
