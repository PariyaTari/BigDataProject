from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import io
import tensorflow as tf
import os
import numpy as np
import keras


app = FastAPI()


# CNN:


# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
model_path = 'cifar10_model.h5'

if not os.path.exists(model_path):
    # Load CIFAR-10 dataset from local directory
    def load_local_cifar10(path):
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        train_data = []
        train_labels = []
        for i in range(1, 6):
            batch = unpickle(os.path.join(path, 'data_batch_' + str(i)))
            train_data.append(batch[b'data'])
            train_labels += batch[b'labels']
        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 32, 32, 3), order='F')
        train_labels = np.array(train_labels)

        test_batch = unpickle(os.path.join(path, 'test_batch'))
        test_data = test_batch[b'data'].reshape((10000, 32, 32, 3), order='F')
        test_labels = np.array(test_batch[b'labels'])

        return (train_data, train_labels), (test_data, test_labels)


    (train_images, train_labels), (test_images, test_labels) = load_local_cifar10('./cifar-10-batches-py')
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Define the model with additional Conv2D layers and Dropout
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=70, validation_data=(test_images, test_labels))

    # Save the model
    model.save(model_path)

    # Evaluate the model on test data
    _, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_accuracy}")

else:
    model = load_model(model_path)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image = await file.read()
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the class of the image
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions[0])]

    return JSONResponse(content={"class": predicted_class})


# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
