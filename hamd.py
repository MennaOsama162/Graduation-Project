import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
import mediapipe as mp
from tensorflow import keras
import itertools


model1 = load_model('"C:/Users/menae/Downloads/TEAM/hamed/actionletter.h5')
class_names = {' ': 0,
 'أ': 1,
 'ئ': 2,
 'ال': 3,
 'ب': 4,
 'ة': 5,
 'ت': 6,
 'ث': 7,
 'ج': 8,
 'ح': 9,
 'خ': 10,
 'د': 11,
 'ذ': 12,
 'ر': 13,
 'ز': 14,
 'س': 15,
 'ش': 16,
 'ص': 17,
 'ض': 18,
 'ط': 19,
 'ظ': 20,
 'ع': 21,
 'غ': 22,
 'ف': 23,
 'ق': 24,
 'ك': 25,
 'ل': 26,
 'لا': 27,
 'م': 28,
 'ن': 29,
 'ه': 30,
 'و': 31
,'ي': 32}


def capletter():
    cap = cv2.VideoCapture(0)
    word = ''
    threshold = 0.97  # Set the threshold to 0.6
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        cv2.rectangle(frame, (400,400), (100,100), (0,255,0), 2)
        crop_img = frame[100:400, 100:400]
        # Convert the frame to grayscale
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # Resize the image to the input size of the model
        resized = cv2.resize(gray, (64, 64))
        # Preprocess the image for input to the model
        preprocessed = np.expand_dims(resized, axis=-1)
        preprocessed = preprocessed.astype('float32') / 255.0
        # Make a prediction using the model
        prediction = model1.predict(np.array([preprocessed]))
        # Get the predicted class label and class name
        predicted_label = np.argmax(prediction)
        predicted_prob = np.max(prediction)
        for key,value in class_names.items():
            if predicted_label == value:
                predicted_class=key
        # Check if the predicted probability is above the threshold
        if predicted_prob > threshold:
            word += predicted_class

          

        sentence_label.config(text=predicted_class)
        sentence_label2.config(text=word)
        
        
        
        
        root.update()
        # Draw the predicted class label on the frame
        cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Display the frame
        cv2.imshow('frame', frame)
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()



actions = np.array(["صباحا","موجود","مباشر","الساعه","كام","مصر","السبت","فلوس","مقطع","تذكره","قطار"])
label_map = {label:num for num, label in enumerate(actions)}
# Load the LSTM model
model2 = keras.models.load_model('C:/Users/menae/Downloads/TEAM/hamed')
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def mediapipe_draw(image,results):
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])






def capword(): 
            sequence = []
            threshold = 0.6
            sentence = []
            cap =cv2.VideoCapture(0)

            # Set mediapipe model 
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                 for i in itertools.count():

                    # Read feed
                    ret, frame = cap.read()
                    # Make detections
                    if not ret:
                         break
                    image, results = mediapipe_detection(frame, holistic)
                    print(results)
                    
                    # Draw landmarks
                    mediapipe_draw(image, results)
                    # 2. Prediction logic
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-50:]
                    
                    if len(sequence) == 50:
                        res = model2.predict(np.expand_dims(sequence, axis=0))[0]
                        print(actions[np.argmax(res)])
                        
                        
                    #3. Viz logic
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5: 
                            sentence = sentence[-5:]

                    sentence_str = ' '.join(sentence)
                    sentence_label22.config(text=sentence_str)
                    root.update()
                    # Viz probabilities
                    cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()









from PIL import Image, ImageTk, ImageFilter
import tkinter as tk
from tkinter import ttk
from tkinter import *
root = tk.Tk()
root.geometry("990x540")
root.minsize(120, 1)
root.maxsize(1540, 845)
root.resizable(1,  1)
root.title("Sign Language Detection")

image = PhotoImage(file="C:/Users/menae/Downloads/TEAM/manci.png")

blur = PhotoImage(file="C:/Users/menae/Downloads/TEAM/manci.png")

label = Label(root, image=image)
label.pack(fill=BOTH, expand=YES)





# Create a label and text box to display the sentence output

#output_label333 = tk.Label(root, anchor='w', background="#008040",
#                               compound='left', font="-family {Segoe UI Symbol} -size 18 -weight bold",
#                               foreground="#000000", text='''Sign Language Detection''')
#output_label333.place(relx=0.0, rely=-0.016, height=59, width=902)



# Create a button to start the sign language detection
start_button = tk.Button(root, activebackground="beige", activeforeground="black",
                                  background="#4F656E", compound='left', disabledforeground="#a3a3a3",
                                  foreground="#000000", highlightbackground="#d9d9d9", highlightcolor="black",
                                  pady="0", text='''فتح الكاميرا للحروف الابجدية''' , command=capletter,font=("Raleway", 14))
start_button.place(relx=0.0, rely=0.112, height=74, width=337)



sentence_label = tk.Label(root, anchor='w' ,compound='right', text=''':الحرف المتوقع''',font=("Raleway", 14),background="#A3A4A2")
###################################sentence_label.configure(image=blur, compound="center")
sentence_label.place(relx=0.033, rely=0.24, height=31, width=302)


sentence_label2 = tk.Label(root, anchor='w', compound='right', text=''':الجملة''',font=("Raleway", 14),background="#A3A4A2")
sentence_label2.place(relx=0.0, rely=0.353, height=81, width=844)






# Create a button to start the sign language detection
start_button22= tk.Button(root, activebackground="beige", activeforeground="#008040",
                                    background="#4F656E", compound='left', disabledforeground="#a3a3a3",
                                    foreground="#000000", highlightbackground="#d9d9d9", highlightcolor="black",
                                    pady="0", command=capword,text='''محطة القطار''',font=("Raleway", 14))
start_button22.place(relx=0.0, rely=0.625, height=74, width=337)

sentence_label22 = tk.Label(root, anchor='w',compound='right',text=''' :الجملة''',font=("Raleway", 14), background="#A3A4A2")
sentence_label22.place(relx=0.0, rely=0.785, height=81, width=844 )

root.update()
root.mainloop()

