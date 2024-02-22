import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved Keras model
model = load_model("model.h5")

# Mapping of class indices to class labels
output_class = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Function for waste prediction
def waste_prediction(new_image):
    test_image = image.load_img(new_image, target_size=(224, 224))
    st.image(test_image, caption='Uploaded Image.', use_column_width=True)

    test_image = image.img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    predicted_array = model.predict(test_image)
    predicted_value = output_class[np.argmax(predicted_array)]
    predicted_accuracy = round(np.max(predicted_array) * 100, 2)

    st.write("Your waste material is", predicted_value, "with", predicted_accuracy, "% accuracy.")

    # Display recycling or disposal message based on the predicted label
    if predicted_value == 'cardboard':
        st.write("Recycle cardboard by flattening it and placing it in the recycling bin.")
    elif predicted_value == 'glass':
        st.write("Recycle glass bottles and jars by rinsing them and placing them in the recycling bin.")
    elif predicted_value == 'metal':
        st.write("Recycle metal cans and containers by rinsing them and placing them in the recycling bin.")
    elif predicted_value == 'paper':
        st.write("Recycle paper by placing it in the recycling bin. Avoid soiled or wet paper.")
    elif predicted_value == 'plastic':
        st.write("Recycle plastic bottles and containers marked with recycling symbols.")
    elif predicted_value == 'trash':
        st.write("Dispose of trash in a waste bin. Consider proper waste disposal guidelines.")

# Streamlit app
def main():
    st.title('GARBAGE CLASSIFICATION USING CNN')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        waste_prediction(uploaded_file)

if __name__ == "__main__":
    main()
