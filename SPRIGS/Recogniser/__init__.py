import cv2
import numpy as np
import face_recognition
import pickle

class Recogniser:

    scaleFactor = 6

    def __init__(self, trainer):
        self.trainer = trainer
        pass

    def start(self):
        video_capture = cv2.VideoCapture(0)
        process_this_frame = True

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=1/self.scaleFactor, fy=1/self.scaleFactor)

            # Only process every other frame of video to save time
            if process_this_frame:
                predictions = self.predict(small_frame)

            process_this_frame = not process_this_frame
            self.highlight_faces(frame, predictions)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    def highlight_faces(self, frame, predictions):

        for name, (top, right, bottom, left) in predictions:
            top = top * self.scaleFactor
            right = right * self.scaleFactor
            bottom = bottom * self.scaleFactor
            left = left * self.scaleFactor
            overlay = frame.copy()
            print(type(name))
            # name = name.encode("UTF-8")
            print(name)
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 8, bottom - 6), font, 0.8, (0, 0, 0), 2)
            frame = cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0)

        # Display the resulting image
        cv2.imshow('Video', frame)

    def predict(self, frame, distance_threshold=0.6):
        with open(self.trainer.MODEL_FILE_NAME, 'rb') as file:
            knn_model = pickle.load(file)

        # Load image from array and find face locations
        image = np.array(frame)
        face_locations = face_recognition.face_locations(image)

        # If no faces are found in the image, return an empty result.
        if len(face_locations) == 0:
            return []

        # Find encodings for faces in the test image
        faces_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_model.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_model.predict(faces_encodings), face_locations, are_matches)]

