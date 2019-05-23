from SPRIGS.Recogniser import Recogniser
from SPRIGS.Trainer import Trainer

def main():
    trainer = Trainer()


    trainer.TRAINING_DATA_DIRECTORY = "train/people"
    trainer.ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    trainer.MODEL_FILE_NAME = "model.clf"

    trainer.learn()  # Checks if model already exits call trainer.train(n_neighbors=2) if training data has been updated.

    recogniser = Recogniser(trainer)

    recogniser.start()


if __name__ == "__main__":
    main()