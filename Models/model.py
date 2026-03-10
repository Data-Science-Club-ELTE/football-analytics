from abc import ABC, abstractmethod

class model(ABC):


    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

