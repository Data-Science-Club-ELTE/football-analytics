from .SklearnModel import LogisticRegressionModel
from .GradiantDescentModel import GradianDescentModel
from .DeepLearningModel import DeepLearningModel
from .model import model
from typing import Tuple, Optional, Literal


model_dict = {"GradientDescent": GradianDescentModel,
              "Logistic Regression": LogisticRegressionModel,
              "Deep Learning" : DeepLearningModel}


class XGmodel(model):
    def __init__(self, lr: float = 0.01, num_iters: int = 100,
                 random_init_weight: bool = True, normalize_data: bool = True,
                 regularization: bool = False, Lambda: int = 1, reduce_dimension: bool = False,
                 reduction_size: Optional[int] = None, model_type: Literal["auto", 'GradientDescent', 'Logistic Regression', 'Deep Learning'] = 'auto'):
        
        if model_type == 'auto':
            pass
        else:
            self.model : model = model_dict[model_type](lr = lr, num_iters = num_iters,
                 random_init_weight = random_init_weight, normalize_data = normalize_data,
                 regularization = regularization, Lambda = Lambda, reduce_dimension = reduce_dimension,
                 reduction_size = reduction_size)


    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)


    def predict(self, X):
        predcited_values, probabilities = self.model.predict(X)
        return predcited_values, probabilities
    
    def setup_auto(self):
        pass
