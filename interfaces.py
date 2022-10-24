# This file defines the informal interaces used for the different classes in this project

from tensorflow.keras.models import Model


# @source https://stackoverflow.com/questions/22046369/enforcing-class-variables-in-a-subclass
class ForceRequiredAttributeDefinitionMeta(type):
    def __call__(cls, *args, **kwargs):
        class_object = type.__call__(cls, *args, **kwargs)
        class_object.check_required_attributes()
        return class_object

class Preprocessing_interface(metaclass = ForceRequiredAttributeDefinitionMeta):
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    label_encoder = None
    classes = None
    random_state = None

    def run(self, img: list) -> list:
        """Runs preprocessing for one image"""
        pass

    def run_set(self, dir: str) -> list:
        """Extract imgaes from directory and run preprocessing for all images in the directory"""
        pass

    def split(self, split: float):
        """Splits the preprocessed images into testing and training sets ready to be put into the model"""
        pass

class Model_interface(metaclass = ForceRequiredAttributeDefinitionMeta):
    model = None

    def define_model(self, prep: Preprocessing_interface) -> None:
        """Define a model given a preprocessed dataset"""
        pass

    def train(self) -> None:
        """Trains the model"""
        pass

    def evaluate(self) -> dict:
        """Evaluates model with plotting and returning accuracy and loss in a dictionary {accuracy: decimal, loss: float}"""
        pass