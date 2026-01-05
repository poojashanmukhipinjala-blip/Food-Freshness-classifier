import gradio as gr
import tensorflow as tf
import cv2
import numpy as np

# Load model
model = tf.keras.models.load_model("food_freshness_model.h5")

def predict_food(image):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    if class_index == 0:
        return "✅ Fresh Food"
    else:
        return "❌ Rotten Food"

interface = gr.Interface(
    fn=predict_food,
    inputs=gr.Image(type="numpy", label="Upload Food Image"),
    outputs=gr.Label(label="Prediction"),
    title="🥗 Food Freshness Classifier",
    description="Upload a food image to check whether it is Fresh or Rotten using Deep Learning."
)

interface.launch()
