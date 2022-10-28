# This file defines the informal interaces used for the different classes in this project

from tensorflow.keras.models import Model


# @source https://stackoverflow.com/questions/22046369/enforcing-class-variables-in-a-subclass
class ForceRequiredAttributeDefinitionMeta(type):
    def __call__(cls, *args, **kwargs):
        class_object = type.__call__(cls, *args, **kwargs)
        class_object.check_required_attributes()
        return class_object

class ForceRequiredAttributeDefinition(metaclass=ForceRequiredAttributeDefinitionMeta):
    starting_day_of_week = None

    

class Preprocessing_interface(metaclass = ForceRequiredAttributeDefinitionMeta):
    """
    For Training:
        Execute the 'prep_training' method on the desired training directory to populate the testing 
        and training variables. Pass the object into the training method in the Model_interface
    
    For Evaluation:
        Execute the 'prep_validation' method on the desired validation directory, it will populate the 
        necessary varaibles to use in the evaluate_model function in a Model_interface

    """

    label_encoder = None
    
    random_state = None

    def run(self, img: list) -> list:
        """Runs preprocessing for one image and returns image"""
        pass

    def prep_training(self, split: float):
        """Splits the preprocessed images into testing and training sets ready to be put into the model
        populates the X_train, X_test, y_train, y_test global varaibles
        """
        pass

    def prep_validation(self, dir: str) -> list:
        """Extract imgaes from directory and run preprocessing for all images in the directory
        populates the X_val and y_val global variables with data
        """
        pass

    def save(self, dir: str) -> None:
        """
        Saves all preprocessed files in pickled files in the designated directory
        """

    def check_required_attributes(self):
        if self.label_encoder is None:
            raise NotImplementedError('Subclass must define self.label_encoder attribute.')
        if self.random_state is None:
            raise NotImplementedError('Subclass must define self.random_state attribute.')
    
class Model_interface(metaclass = ForceRequiredAttributeDefinitionMeta):
    

    def define_model(self) -> None:
        """Define a model"""
        pass

    def train(self) -> None:
        """Trains the model
        The preprocessing_interface passed into the function must have run the 'prep_training' function to populate the X_train, X_test,
        y_train, y_test object variables
        
        """
        pass

    def evaluate(self) -> dict:
        """Evaluates model with plotting and returning accuracy and loss in a dictionary {accuracy: decimal, loss: float}
        Thsi preprocessing_interface MUST be different that the one used to train the dataset, otherwise we're evaluating the model based on the
        training data. 

        the preprocessing_interface object must have ran the 'prep_validation' function to create the validation set of data
        """
        pass

    def predict(self, image) -> str:
        """Returns a prediction given a preprocessed image"""
        pass

    def save(self) -> None:
        """saves model artifacts"""
        pass

    def check_required_attributes(self):
        pass