import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def confusion_matrix_to_accuraccies(confusion_matrix, ignore_missing=False):

    confusion_matrix = confusion_matrix.astype(float)
    # sum(0) <- predicted sum(1) ground truth
    total = np.sum(confusion_matrix)
    n_classes, _ = confusion_matrix.shape
    overall_accuracy = np.sum(np.diag(confusion_matrix)) / total

    # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
    N = total
    p0 = np.sum(np.diag(confusion_matrix)) / N
    pc = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / N ** 2
    kappa = (p0 - pc) / (1 - pc)

    recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 1e-12)
    precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + 1e-12)
    f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)

    # Per class accuracy
    cl_acc = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + 1e-12)

    # exclude classes with no training samples
    if ignore_missing:
        # mask for classes with sample points (miminimum of 1 sample point in train or valid)
        present_mask = (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0)) > 0
        recall = recall[present_mask]
        precision = precision[present_mask]
        f1 = f1[present_mask]
        cl_acc = cl_acc[present_mask]

    return overall_accuracy, kappa, precision, recall, f1, cl_acc

class ClassMetric(object):
    def __init__(self, num_classes=2, ignore_index=0):
        self.num_classes = num_classes
        _range = -0.5, num_classes - 0.5
        self.range = np.array((_range, _range), dtype=np.int64)
        self.ignore_index = ignore_index
        self.hist = np.zeros((num_classes, num_classes), dtype=np.float64)

        self.store = dict()

        self.earliness_record = list()

    def _update(self, o, t):
        t = t.flatten()
        o = o.flatten()
        # confusion matrix
        n, _, _ = np.histogram2d(t, o, bins=self.num_classes, range=self.range)

        self.hist += n

    def add(self, stats):
        for key, value in stats.items():

            value = value.data.cpu().numpy()

            if key in self.store.keys():
                self.store[key].append(value)
            else:
                self.store[key] = list([value])

        return dict((k, np.stack(v).mean()) for k, v in self.store.items())

    def update_confmat(self, target, output, ignore_missing=False):
        self._update(output, target)
        return self.accuracy(ignore_missing=ignore_missing)

    def update_earliness(self,earliness):
        self.earliness_record.append(earliness)
        return np.hstack(self.earliness_record).mean()

    def accuracy(self, ignore_missing=False):
        """
        https: // en.wikipedia.org / wiki / Confusion_matrix
        Calculates over all accuracy and per class classification metrics from confusion matrix
        :param confusion_matrix numpy array [n_classes, n_classes] rows True Classes, columns predicted classes:
        :return overall accuracy
                and per class metrics as list[n_classes]:
        """
        confusion_matrix = self.hist

        if type(confusion_matrix) == list:
            confusion_matrix = np.array(confusion_matrix)

        overall_accuracy, kappa, precision, recall, f1, cl_acc = confusion_matrix_to_accuraccies(confusion_matrix, ignore_missing=ignore_missing)

        return dict(
            overall_accuracy=overall_accuracy,
            kappa=kappa,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=cl_acc
        )

class RegressionMetric(object):
    def __init__(self):
        self.rmse_values = []
        self.r2_values = []
        self.store = dict()
    def _update(self, true_values, predicted_values):
        mse = mean_squared_error(true_values, predicted_values)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_values, predicted_values)

        self.rmse_values.append(rmse)
        self.r2_values.append(r2)

    def update_mat(self, true_values, predicted_values):
        self._update(true_values, predicted_values)
        return {
            "rmse": np.mean(self.rmse_values),
            "r2": np.mean(self.r2_values)
        }

    def add(self, stats):
        for key, value in stats.items():

            value = value.data.cpu().numpy()

            if key in self.store.keys():
                self.store[key].append(value)
            else:
                self.store[key] = list([value])

        return dict((k, np.stack(v).mean()) for k, v in self.store.items())