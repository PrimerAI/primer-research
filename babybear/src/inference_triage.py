import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import pipeline
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from nlx_babybear import RFBabyBear
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import pickle as pkl

np.random.seed(0)
import time

class MamabearClassifierSentiment:
    """
    """

    def __init__(
        self,
        model,
        device=0,
        tokenizer=None
    ):

        self.classifier = pipeline("sentiment-analysis", model=model, tokenizer=model, device=device, return_all_scores=True)


    def _run_mamabear(self, texts):
        """Runs mamabear model on the input text file.

        Args:
            texts: Documents sent to mamabear for classification

        Return:
            mama_pred: Prediction probability calculated by mamabear
        """
        labels = []
        for text in texts:
            score = []
            prediction = self.classifier(text)

            for counter in range(3):
                score += [prediction[0][counter]['score']]
            labels += [np.argmax(np.asarray(score))]

        return {'y_mamabear':np.asarray(labels)}

    def __call__(self, doc):
        """Alias for _run_mamabear.

        See _run_mamabear for documentation.
        """
        return self._run_mamabear(doc)


class MamabearClassifier:
    """This class defines the mamabear classifier which is a model from primer_nlx.nn.classifier.
    The current design is for binary classification. In case of multi-classes classification, by
    summing up labels_to_sum_up, the problems is converted to a binary classification problem.

    Attributes:
        labels_to_sum_up: Labels of classes that are summed to represent class 1.
        model: classification model
        language_model: Language model to use, i.e. 'bert' or 'xlnet'
        model_type: The specific variant of the language model, e.g. 'base-uncased'.
            For language_model == 'xlnet', this must be 'base-cased.'
        device: Device for Pytorch, i.e. 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model,
        device=0,
        tokenizer=None
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=device)

    def _run_mamabear(self, texts):
        """Runs mamabear model on the input text file.

        Args:
            texts: Documents sent to mamabear for classification

        Return:
            mama_pred: Prediction probability calculated by mamabear
        """
        mama_pred = []
        for text in texts:
            ner = self.nlp(text)
            if not ner:
                mama_pred += [0]
            else:
                mama_pred += [1]
        return {'y_mamabear':np.asarray(mama_pred)}

    def __call__(self, doc):
        """Alias for _run_mamabear.

        See _run_mamabear for documentation.
        """
        return self._run_mamabear(doc)

class MamabearClassifierEmotion:
    """
    """

    def __init__(
        self,
        model,
        device=0,
        tokenizer=None
    ):
        self.model = model

        self.classifier = pipeline("text-classification",model=model,device=device, return_all_scores=True)

    def _run_mamabear(self, texts):
        """Runs mamabear model on the input text file.

        Args:
            texts: Documents sent to mamabear for classification

        Return:
            mama_pred: Prediction probability calculated by mamabear
        """
        labels = []
        for text in texts:
            score = []
            prediction = self.classifier(text)
            for counter in range(6):
                score += [prediction[0][counter]['score']]
            labels += [np.argmax(np.asarray(score))]
        return {'y_mamabear':np.asarray(labels)}

    def __call__(self, doc):
        """Alias for _run_mamabear.

        See _run_mamabear for documentation.
        """
        return self._run_mamabear(doc)


class TriagedClassifier():
    """Applies inference triage.

    Attributes:
        babybear: babybear model
        mamabear: mamabear model
        metric_threshold: The minimum threshold of the performance metric defined by the user
        metric: The performance metric which is one of the options "accuracy", "precision", "f1_score", "recall
            The default performance metric is "f1_score"
        confidence_th_options: confidence intervals for finding confidence threshold.
            The default value is np.arange(0, 1.02, 0.02)
    """

    def __init__(
        self, task, babybear, mamabear, metric_threshold, metric=None, confidence_th_options=None
    ):
        """"""
        super().__init__()
        self.babybear = babybear
        self.mamabear = mamabear
        self.metric_threshold = metric_threshold
        self.task = task
        self.confidence_th_options = confidence_th_options if len(confidence_th_options)!=0 else np.arange(0,1.005,.005)
        self.metric = metric or "accuracy"

        # confidence threshold to get the performance closest to the metric_threshold
        self.confidence_th = 1
        # Percentage of documents which are not sent to mamabear after applying inference triage
        self.saving = []
        self.performance = []

        self.tot_time = []

    def plot_output(self, saving, accuracy, f1, precision, recall, cut, fold_num):
        output_dict = {}
        output_dict['saving'] = saving
        output_dict['accuracy'] = accuracy
        output_dict['f1'] = f1
        output_dict['precision'] = precision
        output_dict['recall'] = recall
        output_dict['cut'] = cut
        if fold_num=="Dev":
            f = open("../fig/dev_data.pkl",'wb')
            pkl.dump(output_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
            f.close()
        plt.figure()
        plt.plot(100*self.confidence_th_options, np.asarray(saving)/100)
        plt.plot(100*self.confidence_th_options, accuracy)
        plt.plot(100*self.confidence_th_options, f1)
        plt.plot(100*self.confidence_th_options, precision)
        plt.plot(100*self.confidence_th_options, recall)
        plt.legend(['saving', 'accuracy', 'f1', 'precision', 'recall'])

        horiz_line = np.arange(0, 110, 10)
        y_horiz_line = np.zeros(len(horiz_line)) +  self.metric_threshold 
        plt.plot(horiz_line, y_horiz_line, ':', color='k')

        y_horiz_line = np.zeros(len(horiz_line)) +  100*self.confidence_th_options[cut]
        plt.plot(y_horiz_line, horiz_line, ':', color='r')

        plt.title(' with max ' + str(self.metric) + '=' + str(100* accuracy[cut]) + '%, we can save '+ str(saving[cut])+ '%' + '\n'+'The Confidence Threshold: ' + str(self.confidence_th_options[cut]))
        plt.xlim(0, 100)
        plt.ylim(0,1.1)

        plt.xlabel('Threshold')
        plt.ylabel(str(self.metric))
        plt.savefig('../fig/' + str(fold_num) + '.jpg')

    def train(self, doc, labels):
        """Finds the confidence threshold.

        This calculations are based on calculating the performance metric for all the values in
        confidence_th_options. The values closest to the metric_threshold will be reported as the
        confidence_th. Saving is calculated using the corresponding value of confidence_th_options.

        Args:
            doc: Input document
            labels: Labels of input documents

        Returns:
            confidence_th: confidence threshold calculated from the input documents.
                The default value is 1 which means no inference triage will be applied.
        """
        fold_num = 0
        idx_cut = []
        doc, labels = np.asarray(doc), np.asarray(labels)
        n_class=len(np.unique(labels))
        
        performance, p_accuracy, p_f1, p_precision, p_recall, saving = [], [], [], [], [], []
        all_data_size = len(labels)
        train_index = int(0.7*all_data_size)

        X_train, X_test_0 = doc[0:train_index], doc[train_index::]
        y_train, y_test_0 = labels[0:train_index], labels[train_index::]
        self.babybear = RFBabyBear(random_state=0)
        self.babybear.train(X_train, y_train, n_class=n_class)

        test_results = self.babybear(X_test_0)
        prob_val = np.asarray(test_results['probs']).reshape(-1, 1)
        predict_val = test_results['labels'].reshape(-1, 1)

        y_test_0 = np.asarray(y_test_0)

        for conf_th in self.confidence_th_options:
            y_baby_mama = np.copy(y_test_0).reshape(-1, 1)
            if self.task == "ner":
                confident_points = np.where((prob_val >= conf_th) & (predict_val==0))[0]
            else:
                confident_points = np.where(prob_val >= conf_th)[0]
            y_baby_mama[confident_points] = predict_val[confident_points]
            diff = predict_val[confident_points] - y_test_0[confident_points].reshape(-1)
            saving.append(len(confident_points)*100/len(predict_val))
            performance.append(self.evaluate(y_test_0, y_baby_mama, self.metric))
            p_accuracy.append(self.evaluate(y_test_0, y_baby_mama, "accuracy"))
            p_f1.append(self.evaluate(y_test_0, y_baby_mama, "f1_score"))
            p_precision.append(self.evaluate(y_test_0, y_baby_mama, "precision"))
            p_recall.append(self.evaluate(y_test_0, y_baby_mama, "recall"))
            
        cut = self._find_nearest(performance, self.metric_threshold)
        self.plot_output(saving, p_accuracy, p_f1, p_precision, p_recall, cut, fold_num)
        idx_cut.append(cut) 
        fold_num += 1

        self.confidence_th = np.max(self.confidence_th_options[idx_cut])
        self.indx_conf_th = self._find_nearest(self.confidence_th_options, self.confidence_th)

        return self.confidence_th

    @staticmethod
    def _find_nearest(array, value):
        """Finds the element in "array" closest to "value".

        Args:
            array: An array to find the closest element to the "value"
            value: value

        Returns:
            idx: Index of the element in the "array" closest to the "value".
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def evaluate(self, y_test_gt, y_test_pred, metric):
        """Compute classification performance.

        This function supports "accuracy", "precision", "recall", "f1_score" and raises  ValueError
        for any other metric.

        Args:
            y_test_gt: Ground truth labels
            y_test_pred: Predicted labels
            metric: Performance metric of interest. It can be any of these
                four metric: "accuracy", "precision", "recall", "f1_score".
                The default metric is "accuracy"

        Returns:
            calculated_metric[metric]: The performance of the model

        Raises:
            ValueError: If the metric is not one of the "f1_score", "accuracy", "precision", "recall".
        """
        acceptable_metrics = ["f1_score", "accuracy", "precision", "recall"]
        if metric not in acceptable_metrics:
            raise ValueError(
                str(metric)
                + " is not an acceptable metric. \n Please choose one of the following"
                + " metrics: accuracy, precision, recall or f1_score"
            )
        calculated_metric = {}
        calculated_metric["accuracy"] = accuracy_score(y_test_gt, y_test_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test_gt, y_test_pred, average="weighted", zero_division=1
        )
        calculated_metric["precision"] = prec
        calculated_metric["recall"] = rec
        calculated_metric["f1_score"] = f1

        return calculated_metric[metric]

    def _triage_inferences(self, babybear_score, ci):
        """Applies inference triage.

        Identify the indices corresponding to unconfident docs. Unconfident docs are the docs where the babybear
        prediction probability is lower than `confidence_th`.

        Args:
            babybear_score: Prediction probability calculated from babybear model

        Returns:
            ModelResult containing:
                unconfident_idx (np.ndarray): index of unconfident documents
                predict (np.ndarray): prediction probability calculated from babybear model
        """

        unconfident_docs_indx = np.asarray(babybear_score < ci).nonzero()[0]
        return unconfident_docs_indx

    def score(self, doc, y_dev):
        """Predicts lables of docs.

        This function predicts the labels of the input documents. First the prediction
        confidence of all the documents is calculated using babybear model. Then the unconfident
        docs are triaged and sent to mamabear to get the full set of predictions.

        Args:
            doc: testing documents

        Return:
            ModelResult containing:
                unconfident_idx (np.ndarray): index of unconfident documents
                predict (np.ndarray): predicted labels.
                    If the babybear is confident the prediction labels is returned other wise the prediction label is calculated using mamabear.
        """
        # calculate baby_mama_pred
        texts = np.asarray(doc)
        babybear_results = self.babybear(texts)
        
        p_accuracy, p_f1, p_precision, p_recall = [], [], [], []

        baby_mama_pred = []
        all_index = np.arange(0, len(texts), 1)
        for ci in self.confidence_th_options:
            t0 = time.time()
            baby_mama_pred = np.copy(babybear_results['labels'])
            if self.task == "ner":
                confident_points = np.where((babybear_results['probs'] >= ci) & (babybear_results['labels']==0))[0]
            else:
                confident_points = np.where(babybear_results['probs'] >= ci)[0]
            unconfident_docs_indx = np.delete(all_index, confident_points)
            if np.any(unconfident_docs_indx):
                
                mama_predict_class = self.mamabear(texts[unconfident_docs_indx])['y_mamabear']

                
                baby_mama_pred[unconfident_docs_indx] = mama_predict_class
            if np.abs(ci - self.confidence_th) < (self.confidence_th_options[1] - self.confidence_th_options[0]):
                test_unconf = unconfident_docs_indx
                test_pred = baby_mama_pred
            t1 = time.time()
            self.saving.append(len(confident_points)*100/len(baby_mama_pred))
            self.performance.append(self.evaluate(y_dev, baby_mama_pred, self.metric))
            p_accuracy.append(self.evaluate(y_dev, baby_mama_pred, "accuracy"))
            p_f1.append(self.evaluate(y_dev, baby_mama_pred, "f1_score"))
            p_precision.append(self.evaluate(y_dev, baby_mama_pred, "precision"))
            p_recall.append(self.evaluate(y_dev, baby_mama_pred, "recall"))
            self.tot_time.append(t1 - t0)
        print('this is for dev')
        self.plot_output(self.saving, p_accuracy, p_f1, p_precision, p_recall, self.indx_conf_th, 'Dev')
        return {'unconfident_idx':test_unconf,
            'preds':test_pred}
            

    def __call__(self, doc):
        """Alias for score.

        See score for documentation.
        """
        return self.score(doc)
