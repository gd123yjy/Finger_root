import os

import cv2
import numpy as np
import tensorflow as tf

from main import build_model

flags = tf.app.flags
FLAGS = flags.FLAGS

# flags.DEFINE_string('image_path', '/home/yjy/dataset/palmprint_dectection/palmprint_val/IMG_0248_012_LLL_012_100.bmp',
#                     'Path to the image to be predicted.')
flags.DEFINE_string('image_dir', '/home/yjy/dataset/palmprint_dectection/LHand/palmvein_val',  # palmprint_val
                    'Path to the where the images are.')
flags.DEFINE_string('image_save_dir', '/home/yjy/dataset/palmprint_dectection/LHand/palmprint_predict',
                    'Path to where the images to be saved.')
flags.DEFINE_string('roi_save_dir',
                    '/home/yjy/PycharmProjects/firstAI/Finger_roots_Eager/datasets/LHand/palmprint_trainval',
                    'Path to where the roi.txt to be saved.')

img_h = int(480)
img_w = int(640)

divided_factor = 255.0


def pre_process(m_image):
    # result = cv2.resize(src=m_image, dsize=None, fx=1, fy=1)
    m_image /= divided_factor
    return m_image


def post_process(m_coordinates):
    return m_coordinates


if __name__ == '__main__':
    m_max = 0
    m_model_name = FLAGS.model_name

    m_model, _ = build_model()

    try:
        if FLAGS.model_name is None:
            for x in os.listdir(FLAGS.model_dir):
                if m_max < int(x[8:10]):
                    m_model_name = x
                    m_max = int(x[8:10])
        # m_model = tf.keras.models.load_model(filepath=os.path.join(FLAGS.model_dir, m_model_name))
        m_model.load_weights(filepath=os.path.join(FLAGS.model_dir, m_model_name))
    except tf.errors.NotFoundError:
        print('model file cannot be found')

    roi_save_file = open(
        os.path.join(FLAGS.roi_save_dir, "figCon_nn.txt"), 'w')
    files = os.listdir(FLAGS.image_dir)

    test_images = np.zeros((1, img_h, img_w, 3))
    for name in files:
        image = cv2.imread(os.path.join(FLAGS.image_dir, name))
        test_images[0] = pre_process(image)

        coordinates = m_model.predict(x=test_images, batch_size=1, steps=1)

        coordinates = post_process(coordinates)

        cv2.circle(image, (coordinates[0][0], coordinates[0][1]), 5, (0, 255, 0))
        cv2.circle(image, (coordinates[0][2], coordinates[0][3]), 5, (0, 255, 0))
        cv2.circle(image, (coordinates[0][4], coordinates[0][5]), 5, (0, 255, 0))

        cv2.imwrite(os.path.join(FLAGS.image_save_dir, name + '.jpg'), image)
        roi_save_file.write("%s %d %d %d %d %d %d\n" % (name, coordinates[0][0], coordinates[0][1], coordinates[0][2],
                                                        coordinates[0][3], coordinates[0][4], coordinates[0][5]))

        # tmp = divided_factor * (test_images[0])
        # tmp = np.copy(tmp)
        #
        # cv2.circle(tmp, (coordinates[0][0], coordinates[0][1]), 3, (0, 0, 255))
        # cv2.circle(tmp, (coordinates[0][2], coordinates[0][3]), 3, (0, 0, 255))
        # cv2.circle(tmp, (coordinates[0][4], coordinates[0][5]), 3, (0, 0, 255))
        #
        # cv2.imwrite(os.path.join(FLAGS.image_save_dir, name + '.jpg'), tmp)
    roi_save_file.close()
