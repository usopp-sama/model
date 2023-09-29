#!/usr/bin/env python

import rospy
import cv2
import tensorflow as tf
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

class ArrowSignDetector:
    def __init__(self):
        rospy.init_node('arrow_sign_detector', anonymous=True)

        # Load the TensorFlow SavedModel
        self.model = tf.keras.models.load_model('/path/to/your/model.h5')

        # Load class labels from the labels.txt file
        with open('/path/to/your/labels.txt', 'r') as file:
            self.class_labels = [line.strip() for line in file.readlines()]

        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Create a subscriber for the camera feed
        self.image_subscriber = rospy.Subscriber('/camera_topic', Image, self.image_callback)

        # Create a publisher for the arrow sign analysis result
        self.result_publisher = rospy.Publisher('/arrow_sign_result', String, queue_size=10)

    def image_callback(self, data):
        try:
            # Convert the ROS Image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            # Resize and preprocess the frame
            frame_width = 224
            frame_height = 224
            new_size = (frame_width, frame_height)
            cv_image = cv2.resize(cv_image, new_size)
            # ... Perform any additional preprocessing here ...

            # Perform inference on the preprocessed frame
            predictions = self.model.predict(np.expand_dims(cv_image, axis=0))

            # Get the index of the class with the highest probability
            predicted_class_index = np.argmax(predictions[0])

            # Get the label of the predicted class
            predicted_class = self.class_labels[predicted_class_index]

            # Publish the result
            self.result_publisher.publish(predicted_class)

            # Display the label on the frame (optional)
            cv2.putText(cv_image, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame with the prediction (optional)
            cv2.imshow('Webcam Feed', cv_image)
            cv2.waitKey(1)

        except CvBridgeError as e:
            rospy.logerr(e)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    detector = ArrowSignDetector()
    detector.run()
