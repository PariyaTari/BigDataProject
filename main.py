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



#FCN:

# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, UpSampling2D, Add, Resizing
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.applications.vgg19 import preprocess_input
# import tensorflow_datasets as tfds
#
# AUTOTUNE = tf.data.AUTOTUNE
#
# NUM_CLASSES = 4
# INPUT_HEIGHT = 224
# INPUT_WIDTH = 224
# LEARNING_RATE = 1e-3
# WEIGHT_DECAY = 1e-4
# EPOCHS = 20
# BATCH_SIZE = 32
# MIXED_PRECISION = True
# SHUFFLE = True
#
# # Load dataset
# (train_ds, valid_ds, test_ds), info = tfds.load(
#     "oxford_iiit_pet",
#     split=["train[:85%]", "train[85%:]", "test"],
#     as_supervised=True,
#     with_info=True
# )
#
# # Preprocess and resize data
# def preprocess_data(image, segmentation_mask):
#     image = tf.image.resize(image, (INPUT_HEIGHT, INPUT_WIDTH))
#     segmentation_mask = tf.image.resize(segmentation_mask, (INPUT_HEIGHT, INPUT_WIDTH))
#     image = preprocess_input(image)
#     return image, segmentation_mask
#
# def build_model():
#     input_layer = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))
#
#     vgg_model = tf.keras.applications.VGG19(include_top=True, weights="imagenet")
#     fcn_backbone = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)
#     fcn_backbone.trainable = False
#
#     x = fcn_backbone(input_layer)
#
#     dense_convs = [
#         Conv2D(4096, (7, 7), activation='relu', padding='same', use_bias=False),
#         Dropout(0.5),
#         Conv2D(4096, (1, 1), activation='relu', padding='same', use_bias=False),
#         Dropout(0.5)
#     ]
#     dense_convs = Sequential(dense_convs)
#     dense_convs.trainable = False
#
#     x = dense_convs(x)
#
#     pool5 = Conv2D(NUM_CLASSES, (1, 1), activation='relu', padding='same')(x)
#
#     fcn32s_conv_layer = Conv2D(NUM_CLASSES, (1, 1), activation='softmax', padding='same')(pool5)
#     fcn32s_upsampling = UpSampling2D(size=(32, 32), interpolation='bilinear')(fcn32s_conv_layer)
#
#     fcn32s_model = Model(inputs=input_layer, outputs=fcn32s_upsampling)
#
#     return fcn32s_model
#
# # Compile and train model
# model = build_model()
# model.compile(
#     optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[
#         tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES),
#         tf.keras.metrics.SparseCategoricalAccuracy()
#     ]
# )
#
# model.fit(train_ds.map(preprocess_data).shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE),
#           epochs=EPOCHS,
#           validation_data=valid_ds.map(preprocess_data).shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE))
#
# # Save the model
# model.save('fcn32s_model.h5')
#
#
# # Check if the model file exists and load the model
# model_path = 'fcn32s_model.h5'
# if os.path.exists(model_path):
#     fcn_model = keras.models.load_model(model_path, compile=False)
# else:
#     raise FileNotFoundError(f"Model file {model_path} not found. Please train and save the model first.")
#
# # Set global mixed precision policy
# policy = keras.mixed_precision.Policy("mixed_float16")
# keras.mixed_precision.set_global_policy(policy)
#
# # Define input dimensions and number of classes
# INPUT_HEIGHT = 224
# INPUT_WIDTH = 224
# NUM_CLASSES = 4
#
#
# def preprocess_image(image_bytes):
#     """Preprocess the uploaded image to the required format."""
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     image = image.resize((INPUT_WIDTH, INPUT_HEIGHT))
#     image_array = np.array(image)
#     image_array = keras.applications.vgg19.preprocess_input(image_array)
#     return image_array
#
#
# def predict_segmentation(image_array):
#     """Predict segmentation mask for the input image array."""
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#     pred_mask = fcn_model.predict(image_array, verbose=0)
#     pred_mask = np.argmax(pred_mask, axis=-1)
#     return pred_mask[0]  # Remove batch dimension
#
#
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     if file.content_type.startswith('image/'):
#         # Read the file bytes
#         image_bytes = await file.read()
#
#         # Preprocess the image
#         image_array = preprocess_image(image_bytes)
#
#         # Predict segmentation mask
#         pred_mask = predict_segmentation(image_array)
#
#         # Convert the prediction to a list for JSON response
#         pred_mask_list = pred_mask.tolist()
#
#         return JSONResponse(content={"segmentation_mask": pred_mask_list})
#     else:
#         return JSONResponse(content={"error": "Invalid file type. Please upload an image file."}, status_code=400)
#

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
