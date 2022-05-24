import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import tensorflow_hub as hub
# ? fix sif
# from primer_nlx.embedding.SIF import SIF
# from primer_nlx.utils.modelresult import ModelResult
# from primer_nlx.zoo import ZooAnimal
# np.random.seed(0)
n_estimators = [int(x) for x in range(200,2000,200)]
grid = {'n_estimators': n_estimators,
        'min_child_weight': range(1,6,2),
        'gamma': [i/10.0 for i in range(0,5)],
        'subsample': [i/10.0 for i in range(6,10)],
        'colsample_bytree': [i/10.0 for i in range(6,10)],
        'max_depth': [3, 6, 9],
        'opt':[1e-5, 1e-2, 0.1, 1, 100],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
        }
args = {'objective': 'binary:logistic', 'nthread':4,
        'scale_pos_weight':1,'seed':27, 'n_jobs': 4, 'use_label_encoder': False}

class RFBabyBear():
    """This class defines the babybear classifier which is sklearn.RandomForestClassifier.

    Attributes:
        language_model: The method to convert the document into a vector.
        The default method is SIF
        kwargs: Optional keyword arguments for the RandomForestClassifier.
    """

    def __init__(self, language_model=None, **rf_kwargs):
        """Inits SampleClass with language_model and keyword arguments."""
        super().__init__()
        self.rf_kwargs = rf_kwargs
        # self.language_model = language_model or SIF("latest-small")
        self.language_model = language_model or hub.load("/Users/leilakhalili/Downloads/universal-sentence-encoder_4")
        self.model = RandomizedSearchCV(XGBClassifier(**self.rf_kwargs), grid)

    def train(self, doc, labels, n_class=2):
        """This function trains babybear.

        Args:
            doc: The training documents
            labels: The lables of the training documents which are human-labeled data
            or provided by papabear.

        Returns:
            ModelResult containing:
                probs (np.ndarray): The prediction probability
                labels (np.ndarray): The prediction labels

        Raises:
            ValueError: If training dataset misses some of the classes.
        """
        doc = list(doc)
        if n_class != len(np.unique(labels)):
            raise ValueError(
                "This solver needs samples of at least " + str(n_class) + " classes in the data, "
                "but the data contains only " + str(len(np.unique(labels))) + " class"
            )
        return self.model.fit(self.language_model(doc).numpy(), labels)

    def score(self, doc):
        """This function predicts using the babybear classifier.

        Args:
            doc: The testing documents

        Returns:
            ModelResult containing:
                probs (np.ndarray): The prediction probability
                labels (np.ndarray): The prediction labels
        """
        doc = list(doc)
        probs = []
        prob_pred = self.model.predict_proba(self.language_model(doc).numpy())
        labels = self.model.predict(self.language_model(doc).numpy())
        for j in range(len(labels)):
            probs += [prob_pred[j][labels[j]]]
        # probs = prob_pred[:, 0]
        
        return {'labels':np.asarray(labels), "probs":np.asarray(probs)}

    def __call__(self, doc):
        # Inference Triage is based on the prediction_probabilties (ModelResults.probs)
        # ModelResults.probs > confidence_th will not be sent to papabear
        return self.score(doc)
