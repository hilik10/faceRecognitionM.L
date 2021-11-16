from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import time
import cv2
import os
# תןספות
import base64

from flask import Flask, render_template, request, jsonify
from flask.wrappers import Request
from werkzeug.exceptions import RequestHeaderFieldsTooLarge
from werkzeug.utils import send_file
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)
def url_to_image(url):
	
  encoded_data = url.split(',')[1]
  nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
  image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  return image


@app.route('/faceRecognition', methods=['POST'])
def getInquiries():
   url=request.json['_imageAsDataUrl']
   image = url_to_image(url)
   result = faceRecognition(image)
   print(result)
   return jsonify(result) 
  #  return jsonify  (request.json['_imageAsDataUrl'])
    


    




def faceRecognition(faceRecognition):
  args = {
      "model": "./liveness.model",
      "le": "./le.pickle",
      "detector": "./face_detector",
      "confidence": 0.5
  }
  print("[INFO] loading face detector...")
  protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
  modelPath = os.path.sep.join([args["detector"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
  net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
  print("[INFO] loading liveness detector...")
  model = load_model(args["model"])
  le = pickle.loads(open(args["le"], "rb").read())
  print("[INFO] starting video stream...")
  # vs = VideoStream(src=0).start()
  # time.sleep(2.0)
  while True:
      # frame = vs.read()
      frame = imutils.resize(faceRecognition, width=600)
      (h, w) = frame.shape[:2]
      blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                  (300, 300), (104.0, 177.0, 123.0))
      net.setInput(blob)
      detections = net.forward()
      for i in range(0, detections.shape[2]):
          confidence = detections[0, 0, i, 2]
          if confidence > args["confidence"]:
              box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
              (startX, startY, endX, endY) = box.astype("int")
              startX = max(0, startX)
              startY = max(0, startY)
              endX = min(w, endX)
              endY = min(h, endY)
              face = frame[startY:endY, startX:endX]
              face = cv2.resize(face, (32, 32))
              face = face.astype("float") / 255.0
              face = img_to_array(face)
              face = np.expand_dims(face, axis=0)
              preds = model.predict(face)[0]
              j = np.argmax(preds)
              label = le.classes_[j]
              if label=="real":



                  args = {
                      "detector": "./face_detection_model",
                      "embedding-model": "./openface_nn4.small2.v1.t7",
                      "recognizer": "./recognizer.pickle",
                      "le": "./le.picklee",
                      "confidence": 0.5
                  }
                  print("[INFO] loading face detector...")
                  protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
                  modelPath = os.path.sep.join([args["detector"],
                                              "res10_300x300_ssd_iter_140000.caffemodel"])
                  detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
                  print("[INFO] loading face recognizer...")
                  embedder = cv2.dnn.readNetFromTorch(args["embedding-model"])
                  recognizer = pickle.loads(open(args["recognizer"], "rb").read())
                  le = pickle.loads(open(args["le"], "rb").read())

                  image = imutils.resize(frame, width=600)
                  (h, w) = image.shape[:2]
                  imageBlob = cv2.dnn.blobFromImage(
                      cv2.resize(image, (300, 300)), 1.0, (300, 300),
                      (104.0, 177.0, 123.0), swapRB=False, crop=False)
                  detector.setInput(imageBlob)
                  detections = detector.forward()
                  for i in range(0, detections.shape[2]):
                      confidence = detections[0, 0, i, 2]
                      if confidence > args["confidence"]:
                          
                          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                          (startX, startY, endX, endY) = box.astype("int")
                          face = image[startY:endY, startX:endX]
                          (fH, fW) = face.shape[:2]
                          if fW < 20 or fH < 20:
                              continue
                          faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                                          (0, 0, 0), swapRB=True, crop=False)
                          embedder.setInput(faceBlob)
                          vec = embedder.forward()
                          preds = recognizer.predict_proba(vec)[0]
                          j = np.argmax(preds)
                          proba = preds[j]
                          name = le.classes_[j]
                          label = "{}: {:.2f}%".format(name, proba * 100)
                          y = startY - 10 if startY - 10 > 10 else startY + 10
                          cv2.rectangle(image, (startX, startY), (endX, endY),
                                      (0, 0, 255), 2)
                          cv2.putText(image, label, (startX, y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                          # cv2.imshow("Image", image)
              elif label=="fake":
                label = "{}: {:.4f}".format(label, preds[j])
                cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
                # cv2.imshow("Frame", frame)
      # key = cv2.waitKey(1) & 0xFF
      # if key == ord("q"):
      #     break
      # cv2.destroyAllWindows()
      return label
  # vs.stop()
if __name__ == "__main__":
    from waitress import serve
    HOST = 'localhost'
    PORT = 8080
    serve( app, host='localhost', port=8080)