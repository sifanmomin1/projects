#!/usr/bin/env python3
"""
Image Classification using CNN with TensorFlow/Keras
- Model definition
- Training pipeline
- Prediction function
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_simple_cnn(input_shape=(224, 224, 3), num_classes=10):
    """
    Create a simple Convolutional Neural Network for image classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        A compiled Keras model
    """
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Create a transfer learning model using MobileNetV2 as the base model.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        A compiled Keras model
    """
    # Load the pre-trained model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_dir, validation_dir, batch_size=32, epochs=10, image_size=(224, 224)):
    """
    Train the model using data from the specified directories.
    
    Args:
        model: The compiled Keras model
        train_dir: Directory containing training data
        validation_dir: Directory containing validation data
        batch_size: Batch size for training
        epochs: Number of epochs to train
        image_size: Size to resize images to
        
    Returns:
        Training history
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    
    return history

def predict_image(model, image_path, image_size=(224, 224), class_names=None):
    """
    Make a prediction for a single image.
    
    Args:
        model: The trained Keras model
        image_path: Path to the image file
        image_size: Size to resize the image to
        class_names: List of class names
        
    Returns:
        Predicted class and confidence
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=image_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class and confidence
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    if class_names:
        predicted_class = class_names[predicted_class_index]
    else:
        predicted_class = f"Class {predicted_class_index}"
    
    return predicted_class, confidence

def main():
    """
    Example usage of the image classification functions.
    """
    print("Image Classification Example")
    print("=" * 30)
    
    # Define parameters
    input_shape = (224, 224, 3)
    num_classes = 10
    batch_size = 32
    epochs = 5
    
    # Create a model
    print("\nCreating model...")
    model = create_simple_cnn(input_shape, num_classes)
    model.summary()
    
    # Print instructions for using the model
    print("\nTo train this model, you would need to:")
    print("1. Prepare your dataset in the following structure:")
    print("   train_dir/")
    print("   ├── class_1/")
    print("   ├── class_2/")
    print("   └── ...", end="\n\n")
    print("2. Call train_model(model, train_dir, validation_dir)")
    print("3. Save the model using model.save('model_name.h5')")
    print("4. Load the model later with tf.keras.models.load_model('model_name.h5')")
    
    print("\nTo make predictions:")
    print("predict_image(model, 'path_to_image.jpg', class_names=['class1', 'class2', ...])")

if __name__ == "__main__":
    # Check if TensorFlow is installed
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        main()
    except ImportError:
        print("TensorFlow is not installed. Please install it with:")
        print("pip install tensorflow")
        print("For more information, visit: https://www.tensorflow.org/install")
