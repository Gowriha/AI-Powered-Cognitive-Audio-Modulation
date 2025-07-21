from deepface import DeepFace

def detect_emotion_from_frame(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion.capitalize()  # Match keys like 'Happy', 'Sad', etc.
    except Exception as e:
        print("Error in emotion detection:", e)
        return "Neutral"
