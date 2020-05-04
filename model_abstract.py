import abc

class Model_Abstract(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, helper, model, name = 'Model'):
        self.helper = helper
        self.model = model
        self.name = name
        self.helper.models.append(self)

    @abc.abstractmethod
    def fit(self, X, Y):
        """implement in subclass"""

    @abc.abstractmethod
    def predict(self, X):
        """implement in subclass"""

    @abc.abstractmethod
    def evaluate_model(self, X, Y):
        """implement in subclass"""
