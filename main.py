import cv2
import dlib
import winsound
import numpy as np
import tensorflow as tf

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    clahe_image = clahe.apply(gray)
    return clahe_image

# Define a function to preprocess the image
def preprocess_image(img_array):
    new_array = cv2.resize(img_array, (80, 80))  # Resize to 80x80
    X_input = np.array(new_array).reshape(-1, 80, 80, 1)  # Add batch dimension
    X_input = X_input / 255.0  # Normalize
    return X_input

# Function to detect eyes using Dlib
def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    eye_regions = []
    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye_start, left_eye_end = 36, 41
        right_eye_start, right_eye_end = 42, 47
        
        left_eye_pts = landmarks.parts()[left_eye_start:left_eye_end+1]
        right_eye_pts = landmarks.parts()[right_eye_start:right_eye_end+1]
        
        left_eye_x, left_eye_y, left_eye_w, left_eye_h = cv2.boundingRect(np.array([(pt.x, pt.y) for pt in left_eye_pts]))
        right_eye_x, right_eye_y, right_eye_w, right_eye_h = cv2.boundingRect(np.array([(pt.x, pt.y) for pt in right_eye_pts]))
        
        left_eye_region = (left_eye_x, left_eye_y, left_eye_x + left_eye_w, left_eye_y + left_eye_h)
        right_eye_region = (right_eye_x, right_eye_y, right_eye_x + right_eye_w, right_eye_y + right_eye_h)
        
        eye_regions.append(left_eye_region)
        eye_regions.append(right_eye_region)
    
    return eye_regions


model = tf.keras.models.load_model('drowsiness_detection_model.h5')

# Parameters for beep sound
frequency = 2500  # Set frequency to 2500 Hz
duration = 500    # Set duration to 500 ms (0.5 seconds)
best_threshold = 0.5

# Offset percentages for eye region adjustment
offsetPercentageW = 3
offsetPercentageH = 3

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Detect eyes using Dlib
        eye_regions = detect_eyes(frame)
        predictions = []
        
        # Draw rectangles around the eyes on the frame
        for (x1, y1, x2, y2) in eye_regions:
            offsetW = (offsetPercentageW / 100) * x2
            x1 = int(x1 - offsetW * 2)
            x2 = int(x2 + offsetW * 2)
            offsetH = (offsetPercentageH / 100) * y2
            y1 = int(y1 - offsetH * 3)
            y2 = int(y2 + offsetH * 3.5)
            
            median_filtered_img = cv2.medianBlur(frame, 7)
            brightened_face_roi = cv2.convertScaleAbs(median_filtered_img, alpha=1.5, beta=0)
            clahe_image = apply_clahe(brightened_face_roi)

            # Extract the eye region and preprocess
            eye_region = clahe_image[y1:y2, x1:x2]
            if eye_region.size == 0:  # Check if eye region is empty
                continue

            # Preprocess the extracted eye region
            processed_img = preprocess_image(eye_region)

            # Make prediction
            prediction = model.predict(processed_img) # Extracting the first value of the prediction
            # Display prediction status
            if prediction >= best_threshold:
                cv2.putText(frame, "Awake", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Sleepy", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            predictions.append(1 if prediction >= best_threshold else 0)

        # If both eyes are detected as "sleepy", sound an alarm
        if len(predictions) == 2 and sum(predictions) == 0:
            winsound.Beep(frequency, duration)

        # Display the frame
        cv2.imshow('Drowsiness Detection', frame)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()