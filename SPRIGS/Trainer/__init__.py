import os
import face_recognition
import math
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn import neighbors
import pickle

class Trainer:

    def __init__(self):
        self.TRAINING_DATA_DIRECTORY = ""
        self.ALLOWED_EXTENSIONS = {}
        self.MODEL_FILE_NAME = ""
        self.KNN_ALGORITHM = "ball_tree"
        self.WEIGHTS = "distance"
        self.VERBOSE = False

    def learn(self):
        self.check_if_exists()

    def train(self, n_neighbors=None):
        X = []
        y = []

        # Loop through each data in the training set
        for person_directory in os.listdir(self.TRAINING_DATA_DIRECTORY):
            if not os.path.isdir(os.path.join(self.TRAINING_DATA_DIRECTORY, person_directory)):
                continue

            # Loop through each training image for the current person
            for image_path in image_files_in_folder(os.path.join(self.TRAINING_DATA_DIRECTORY, person_directory)):
                image = face_recognition.load_image_file(image_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if self.VERBOSE:
                        print("Image {} not suitable for training: {}".format(image_path, "Didn't find a face" if len(
                            face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    y.append(person_directory)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if self.VERBOSE:
                print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        knn_classifier = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm=self.KNN_ALGORITHM,
            weights=self.WEIGHTS
        )
        knn_classifier.fit(X, y)

        # Save the trained KNN classifier
        if self.MODEL_FILE_NAME is not None:
            with open(self.MODEL_FILE_NAME, 'wb') as f:
                pickle.dump(knn_classifier, f)

        return knn_classifier

    def check_if_exists(self):
        exists = os.path.isfile(self.MODEL_FILE_NAME)
        if not exists:
            print("Training KNN classifier...")
            self.train(n_neighbors=2)
            print("Training complete!")
