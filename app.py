from tensorflow.keras.models import load_model


model = load_model('hand_gesture_cnn_84.keras') 

gesture_classes = {
    0: '01_palm',
    1: '02_l',
    2: '03_fist',
    3: '04_fist_moved',
    4: '05_thumb',
    5: '06_index',
    6: '07_ok',
    7: '08_palm_moved',
    8: '09_c',
    9: '10_down'
}

import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI) for hand (adjust as needed)
    roi = frame[100:300, 100:300]  # y1:y2, x1:x2

    # Preprocess ROI
    img = cv2.resize(roi, (64, 64))  # Match training size
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)    # Add batch dimension

    # Predict gesture
    preds = model.predict(img)
    class_id = np.argmax(preds[0])
    gesture = gesture_classes[class_id]

    # Display results
    cv2.rectangle(frame, (100, 100), (300, 300), (0,255,0), 2)
    cv2.putText(frame, f'Gesture: {gesture}', (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


