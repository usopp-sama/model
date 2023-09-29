import cv2
import tensorflow as tf
import numpy as np

# Load the TensorFlow SavedModel
model = tf.keras.models.load_model('/home/go4av05/Desktop/model/keras_model_x.h5')  # Replace with the actual path to your model folder

# Load class labels from the labels.txt file
with open('/home/go4av05/Desktop/model/labels_x.txt', 'r') as file:
    class_labels = [line.strip() for line in file.readlines()]

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 represents the default camera (you may need to change it if you have multiple cameras)
# Set the desired frame size (224x224)
frame_width = 224
frame_height = 224
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        break
  # Resize the frame to (224, 224)
    new_size = (224, 224)
    frame = cv2.resize(frame, new_size)

    # Crop the frame to (224, 224, 3)
    crop_start_x = (frame.shape[1] - frame_width) // 2
    crop_start_y = (frame.shape[0] - frame_height) // 2
    frame = frame[crop_start_y:crop_start_y + frame_height, crop_start_x:crop_start_x + frame_width]

    # Preprocess the frame (resize, normalize, etc.) to match the input requirements of your model
    # Replace this with your specific preprocessing code

    # Perform inference on the preprocessed frame
    predictions = model.predict(np.expand_dims(frame, axis=0))  # Expand dimensions to create a batch of size 1

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])

    # Get the label of the predicted class
    predicted_class = class_labels[predicted_class_index]

    # Display the label on the frame
    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the prediction
    cv2.imshow('Webcam Feed', frame)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
