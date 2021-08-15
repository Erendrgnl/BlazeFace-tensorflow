import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def generate_bboxes_from_landmarks(landmarks):
    """Generating bounding boxes from landmarks.
    inputs:
        landmarks = (M, total_landmarks, [x, y])

    outputs:
        bboxes = (M, [y1, x1, y2, x2])
    """
    padding = 5e-3
    x1 = tf.reduce_min(landmarks[..., 0], -1) - padding
    x2 = tf.reduce_max(landmarks[..., 0], -1) + padding
    y1 = tf.reduce_min(landmarks[..., 1], -1) - padding
    y2 = tf.reduce_max(landmarks[..., 1], -1) + padding
    #
    gt_boxes = tf.stack([y1, x1, y2, x2], -1)
    return tf.clip_by_value(gt_boxes, 0, 1)

def filter_landmarks(landmarks):
    """Filtering landmark from 68 points to 6 points for blazeface.
    inputs:
        landmarks = (M, 68, [x, y])

    outputs:
        filtered_landmarks = (M, 6, [x, y])
    """
    # Left eye
    left_eye_coords = tf.reduce_mean(landmarks[..., 36:42, :], -2)
    # Right eye
    right_eye_coords = tf.reduce_mean(landmarks[..., 42:48, :], -2)
    # Left ear
    left_ear_coords = tf.reduce_mean(landmarks[..., 0:2, :], -2)
    # Right ear
    right_ear_coords = tf.reduce_mean(landmarks[..., 15:17, :], -2)
    # Nose
    nose_coords = tf.reduce_mean(landmarks[..., 27:36, :], -2)
    # Mouth
    mouth_coords = tf.reduce_mean(landmarks[..., 48:68, :], -2)
    return tf.stack([
        left_eye_coords,
        right_eye_coords,
        left_ear_coords,
        right_ear_coords,
        nose_coords,
        mouth_coords,
    ], -2)

def preprocessing(image_data, final_height, final_width, augmentation_fn=None):
    """Image resizing operation handled before batch operations.
    inputs:
        image_data = tensorflow dataset image_data
        final_height = final image height after resizing
        final_width = final image width after resizing

    outputs:
        img = (final_height, final_width, channels)
        gt_boxes = (gt_box_size, [y1, x1, y2, x2])
        gt_landmarks = (gt_box_size, total_landmarks, [x, y])
    """
    img = image_data["image"]
    img = tf.image.convert_image_dtype(img, tf.float32)
    gt_landmarks = tf.expand_dims(image_data["landmarks_2d"], 0)
    gt_boxes = generate_bboxes_from_landmarks(gt_landmarks)
    gt_landmarks = filter_landmarks(gt_landmarks)
    img = tf.image.resize(img, (final_height, final_width))
    if augmentation_fn:
        img, gt_boxes, gt_landmarks = augmentation_fn(img, gt_boxes, gt_landmarks)
    img = (img - 0.5) / 0.5
    return img, gt_boxes, gt_landmarks

def get_data():
    train_split = "train[:80%]"
    train_data, info = tfds.load("the300w_lp", split=train_split, data_dir="~/tensorflow_datasets", with_info=True,batch_size=32)

    train_data = train_data.map(lambda x : preprocessing(x, 128, 128,))
    
    for d in train_data:
        #x  = np.array(d[2])
        #print(type(x))
        #print(x.shape)
        #break
        #yield(d[0],d[1],d[2])
        yield np.array(d[0], dtype=np.float32), 
        [np.array(d[1], dtype=np.float32), np.array(d[2], dtype=np.float32)]

if __name__ == "__main__":
    get_data()

