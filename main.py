import streamlit as st
import cv2
import tempfile
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
from tensorflow.keras.preprocessing.image import load_img, img_to_array

face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Load the trained eye state detection model
model = load_model('models/custmodel.h5')

st.title("Face Drowsiness Detection System")

# Streamlit Menu
choice = st.sidebar.selectbox("MY MENU", ("HOME", "IMAGE", "VIDEO","CAMERA"))

if choice == "HOME":
    st.write("Welcome to the Drowsiness Detection System!")
    st.image("https://www.ecommunity.com/sites/default/files/styles/blog_post_desktop/public/blog-posts/2020-03/thats-it-im-done-picture-id936117884.jpg?itok=6HjtqKSn")



elif choice == "IMAGE":
    file = st.file_uploader("Upload Image")

    if file:
        # Convert uploaded file to OpenCV format
        b = file.getvalue()
        d = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(d, cv2.IMREAD_COLOR)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            left_eye = left_eye_cascade.detectMultiScale(roi_gray)
            right_eye = right_eye_cascade.detectMultiScale(roi_gray)

            left_eye_status = "Open"
            right_eye_status = "Open"

            for (ex, ey, ew, eh) in left_eye:
                left_eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                left_eye_img = cv2.resize(left_eye_img, (24, 24)) / 255.0
                left_eye_img = left_eye_img.reshape(24, 24, -1)
                left_eye_img = np.expand_dims(left_eye_img, axis=0)

                left_eye_pred = np.argmax(model.predict(left_eye_img), axis=-1)
                if left_eye_pred[0] == 0:
                    left_eye_status = "Closed"
                break

            for (ex, ey, ew, eh) in right_eye:
                right_eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                right_eye_img = cv2.resize(right_eye_img, (24, 24)) / 255.0
                right_eye_img = right_eye_img.reshape(24, 24, -1)
                right_eye_img = np.expand_dims(right_eye_img, axis=0)

                right_eye_pred = np.argmax(model.predict(right_eye_img), axis=-1)
                if right_eye_pred[0] == 0:
                    right_eye_status = "Closed"
                break

            # Drowsiness Detection Logic
            if left_eye_status == "Closed" and right_eye_status == "Closed":
                color = (0, 0, 255)  # Red for Drowsy
                label = "Drowsy"
            else:
                color = (0, 255, 0)  # Green for Awake
                label = "Awake"

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        st.image(img, channels="BGR", width=400)

elif choice == "VIDEO":
    st.subheader("Upload a Video for Drowsiness Detection")
    file = st.file_uploader("Upload Video")
    window = st.empty()

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        vid = cv2.VideoCapture(tfile.name)

        while vid.isOpened():
            flag, frame = vid.read()
            if not flag:
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                left_eye = left_eye_cascade.detectMultiScale(roi_gray)
                right_eye = right_eye_cascade.detectMultiScale(roi_gray)

                left_eye_status, right_eye_status = "Open", "Open"

                # Check left eye status if eyes are detected
                if len(left_eye) > 0:
                    for (ex, ey, ew, eh) in left_eye:
                        left_eye_img = roi_gray[ey:ey + eh, ex:ex + ew]
                        left_eye_img = cv2.resize(left_eye_img, (24, 24)) / 255.0
                        left_eye_img = left_eye_img.reshape(24, 24, -1)
                        left_eye_img = np.expand_dims(left_eye_img, axis=0)

                        left_eye_pred = np.argmax(model.predict(left_eye_img), axis=-1)
                        if left_eye_pred[0] == 0:  # Closed
                            left_eye_status = "Closed"
                        break  # Exit loop once the first eye is processed

                # Check right eye status if eyes are detected
                if len(right_eye) > 0:
                    for (ex, ey, ew, eh) in right_eye:
                        right_eye_img = roi_gray[ey:ey + eh, ex:ex + ew]
                        right_eye_img = cv2.resize(right_eye_img, (24, 24)) / 255.0
                        right_eye_img = right_eye_img.reshape(24, 24, -1)
                        right_eye_img = np.expand_dims(right_eye_img, axis=0)

                        right_eye_pred = np.argmax(model.predict(right_eye_img), axis=-1)
                        if right_eye_pred[0] == 0:  # Closed
                            right_eye_status = "Closed"
                        break  # Exit loop once the first eye is processed

                # Determine the color based on eye status
                if left_eye_status == "Closed" or right_eye_status == "Closed":
                    color = (0, 0, 255)  # Red for closed eyes
                    label = "Drowsy"
                else:
                    color = (0, 255, 0)  # Green for awake (eyes open)
                    label = "Awake"

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 7)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 7)

            # Show frame in Streamlit
            window.image(frame, channels='BGR')

        vid.release()
elif choice=="CAMERA":
    
    
    st.title("Drowsiness Detection System")

    mixer.init()
    alarm_sound = mixer.Sound('alarm.wav')

    st.title("Drowsiness Detection System")

    # Initialize Session State
    if "camera_running" not in st.session_state:
        st.session_state["camera_running"] = False

    if "stop_signal" not in st.session_state:
        st.session_state["stop_signal"] = False

    # Camera Start Input
    start_input = st.text_input("Enter '0' to Start Camera:", "")

    if start_input == "0":
        st.session_state["camera_running"] = True
        st.session_state["stop_signal"] = False

    if st.button("Stop Camera"):
        st.session_state["stop_signal"] = True
        st.session_state["camera_running"] = False

    if st.session_state["camera_running"]:
        counter = 0
        time_inactive = 0
        alarm_triggered = False
        thick = 2
        last_time_closed = time.time()

        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            st.error("Cannot access the webcam!")
        else:
            run = st.empty()

            while st.session_state["camera_running"]:
                ret, frame = capture.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                left_eye_status, right_eye_status = "Open", "Open"
                drowsy_detected = False

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    left_eye = left_eye_cascade.detectMultiScale(roi_gray)
                    right_eye = right_eye_cascade.detectMultiScale(roi_gray)

                    # **Detect Eye Closure**
                    if len(left_eye) == 0:
                        left_eye_status = "Closed"
                    if len(right_eye) == 0:
                        right_eye_status = "Closed"

                    # **Drowsiness Detection Logic**
                    if left_eye_status == "Closed" and right_eye_status == "Closed":
                        time_inactive += 1  # Increase counter
                        drowsy_detected = True
                        last_time_closed = time.time()
                    else:
                        time_inactive = max(0, time_inactive - 1)  # Decrease counter safely

                    # **Alarm Logic**
                    if time_inactive > 10:
                        if not alarm_triggered:
                            alarm_sound.play(-1)  # Play Alarm Continuously
                            alarm_triggered = True
                    else:
                        if alarm_triggered:
                            alarm_sound.stop()  # Stop Alarm
                            alarm_triggered = False

                    # **Draw Bounding Box**
                    color = (0, 0, 255) if drowsy_detected else (0, 255, 0)
                    label = "Drowsy" if drowsy_detected else "Awake"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thick)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # **Time Inactive Message**
                    cv2.putText(frame, f'Inactive Time: {time_inactive}s', (10, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # **Display in Streamlit**
                run.image(frame, channels='BGR')

                # **Stop Camera Based on Input**
                if st.session_state["stop_signal"]:
                    break

            capture.release()
            cv2.destroyAllWindows()
