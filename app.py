from flask import Flask, render_template, Response, jsonify, request
from keras.models import load_model
import cv2
import numpy as np
import traceback
import requests
import json

app = Flask(__name__)

# ------------------ EMOTION DETECTION SETUP ------------------ #
model = load_model('model.h5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
current_emotion = "Neutral"

# ------------------ OLLAMA MISTRAL SETUP ------------------ #
OLLAMA_API_URL = #API
OLLAMA_MODEL = "mistral"

# ------------------ EMOTION DETECTION FUNCTION ------------------ #
def detect_emotion():
    global current_emotion
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (64, 64))
            face = face.reshape(1, 64, 64, 1) / 255.0

            prediction = model.predict(face)
            label_index = np.argmax(prediction)
            current_emotion = emotion_labels[label_index]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, current_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ------------------ ROUTES ------------------ #
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion', methods=['POST'])
def get_emotion():
    return jsonify({'emotion': current_emotion})

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get("message", "")
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": user_input,
            "system": "You are a sweet and supportive mental health advisor. Be empathetic, warm, and comforting in your replies.",
            "stream": False
        }

        response = requests.post(OLLAMA_API_URL, data=json.dumps(payload))
        result = response.json()

        if 'response' not in result or not result['response'].strip():
            return jsonify({
                'reply': "I'm really sorry you're feeling this way. You're not alone, and I'm here to listen. Try to take a deep breath, and if you feel up to it, tell me more about what's going on."
            })

        reply = result['response'].strip()
        return jsonify({'reply': reply})

    except Exception as e:
        print("Chatbot Error:", e)
        traceback.print_exc()
        return jsonify({
            'reply': "Sorry, I'm having trouble responding right now."
        })


# ------------------ MAIN ------------------ #
if __name__ == "__main__":
    app.run(debug=True, port=5000)
