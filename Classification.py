import numpy as np
import tensorflow as tf
from keras.models import load_model


def classify_image(image_path, model_path='D:/neiro/my_model.h5'):
    model = load_model(model_path)

    img_height = 512
    img_width = 512

    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])  # Преобразование в вероятности

    # Определение названий классов
    try:
        class_names = model.class_names
    except AttributeError:
        class_names = ['healthy', 'sick']

    # Вычисление класса с наибольшей вероятностью
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    print(f"This image most likely belongs to {predicted_class} with a {confidence:.2f} percent confidence.")


if __name__ == "__main__":
    while True:  # Бесконечный цикл для постоянного запроса пути изображения ))))))))))
        try:
            image_path = input("Please enter the path to the image (or 'exit' to quit): ")
            if image_path.lower() == 'exit':
                break
            classify_image(image_path)
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
