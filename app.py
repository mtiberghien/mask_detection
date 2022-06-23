from flask import Flask, render_template, Response
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np


app = Flask(__name__)
# Caméra Arduino (l'IP à renseigner dépend du réseau)
camera = cv2.VideoCapture(0)
# Chargement du modèle de détection des masques
json_file = open('./static/models/mask_detector.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Lecture des poids du modèles
loaded_model.load_weights('./static/models/mask_detector.h5')

# Chargement du modèle de détection des visages
face_cascade = cv2.CascadeClassifier('./static/models/face_detector.xml')

print("the mask detector model is loaded successfully !")
target_names = ["with_mask", "no_mask"]


def gen_frames():
    while True:
        # Lecture de la trame
        success, img = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # La détection des visages requiert une image en nuance de gris
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Détection des régions des visages
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Pour chaque visage, on prédit le port ou non d'un masque
            for (x, y, w, h) in faces:
                roi_image = img[y:y + h, x:x + w]
                # L'image est redimensionnée pour correspondre au modèle de prédiction
                color_img = cv2.resize(roi_image, (200, 200))
                color_tensor = np.expand_dims(color_img, axis=0)
                result = loaded_model.predict(color_tensor, verbose=0)
                target_index = np.argmax(result, axis=-1)[0]
                text = target_names[target_index]
                frame_color = ((0, 255, 0) if target_index == 0 else (0, 0, 255))
                cv2.rectangle(img, (x, y), (x + w, y + h), frame_color, 2)
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, frame_color, 2)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
