import cv2
import os
from flask import Flask, request, jsonify
from datetime import date, datetime
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from augmentation_single import augment_images_in_directory
from flask_pymongo import PyMongo
from flask_cors import CORS
# Defining Flask App
app = Flask(__name__)
CORS(app)

port_app = 8000

app.config["MONGO_URI"] = "mongodb://localhost:27017/pep"  # Update with your MongoDB URI

mongo = PyMongo(app)

current_date = datetime.now().strftime("%Y%m%d")

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

def identify_face(facearray):
    try:
        model = tf.keras.models.load_model('static/face_detection_model.keras')
        facearray = np.array(facearray)
        if facearray.ndim == 3:
            facearray = np.expand_dims(facearray, axis=0)  
        facearray = facearray.astype('float32') / 255.0
        facearray = tf.image.resize_with_pad(facearray, 224, 224)  
        predictions = model.predict(facearray)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        class_names = np.load('face_detection_model/class_names.npy')
        
        return class_names[predicted_class_index]
    
    except Exception as e:
        return f"Error during face identification: {str(e)}"

def train_model():
    os.system(f'python {'train_model.py'}')

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    filter = {
        "domain": g_domain,
        "batch": int(g_batch),
        "students.regisno": int(userid),
    }
    
    update = {
        "$set": {
            "students.$.afterPresent": True
        }
    }
    
    result = mongo.db.mains.update_one(filter, update)
    
    if result.matched_count > 0:
        print("Student updated successfully")
    else:
        print("Student not found")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

################## ROUTING FUNCTIONS #########################
g_domain = None
g_batch = None

@app.route('/')
def home():
    # global g_domain, g_batch
    # g_domain = domain
    # g_batch = batch
    if not os.path.isdir('static/faces' + g_domain): 
        os.makedirs('static/faces' + g_domain)
    if not os.path.isdir('static/faces' + g_domain + '/' + g_batch): 
        os.makedirs('static/faces' + g_domain + '/' + g_batch)
    return jsonify({})

## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/' + duser)

    if os.listdir('static/faces/') == []:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except:
        pass
    return jsonify({})

# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start/', methods=['GET', 'POST'])
def start():
    print("received request to take Attendance")

    if 'face_detection_model.keras' not in os.listdir('static'):
        return jsonify({
            "message": 'There is no trained model in the static folder. Please add a new face to continue.'
        }), 400

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"message": "Webcam not accessible"}), 500

    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam")
            break
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face)
            # if identified_person.startswith("Error"):
            if identified_person: print(identified_person)
            else:
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({})

@app.route('/add/', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + newuserid
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"message": "Webcam not accessible"}), 500

    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs * 5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    augment_images_in_directory(userimagefolder)
    print('Training Model')
    train_model()
    return jsonify({})

# Our main function which runs the Flask App
if __name__ == '__main__':
    print(f"Python server is running at {port_app}")
    app.run(debug=True, port=port_app)

