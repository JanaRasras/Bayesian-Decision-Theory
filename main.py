# !!! Must Not Change the Imports !!!
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from TicToc import Timer, timing


def loadDataset(filepath='Data/Pokemon.csv', labelCol='type1') -> (np.ndarray, np.ndarray, LabelEncoder):
    """
    !!! Must Not Change the Content !!!
    This function is used to load dataset
    Data will be scaled from 0 to 1
    :param filepath: csv file path
    :param labelCol: column name
    :return: X: ndarray, y: ndarray, labelEncoder
    """
    data = pd.read_csv(filepath)
    y = data[labelCol].values
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(y)
    X = data.drop(columns=[labelCol]).values
    X = np.nan_to_num(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y, labelEncoder


def loadUtilityMat(matFilePath='Data/UtilityMatrix.csv', matIndexCol='class') -> np.ndarray:
    """
    !!! Must Not Change the Content !!!
    Load utility matrix
    :param matFilePath: matrix file path
    :param matIndexCol: No need to change
    :return: matrix: ndarray
    """
    mat = pd.read_csv(matFilePath, index_col=matIndexCol)
    return mat.values


class Classifier:

    def __init__(self, utilityMat):
        """
        !!! Must Not Change the Content !!!
        :param utilityMat: utility matrix
        """
        self.X = None
        self.y = None
        self.classes = None
        self.meanDict = dict()
        self.priorDict = dict()
        self.cov = None
        self.covInv = None
        self.mvndDenominator = 0
        self.utilityMat = utilityMat

    @staticmethod
    def inv(mat) -> np.ndarray:
        """
        !!! Must Not Change the Content !!!
        Apply svd to calculate the inverse of the matrix
        :param mat: matrix
        :return: inverse of the matrix
        """
        u, s, v = np.linalg.svd(mat, full_matrices=False)
        return np.matmul(v.T * 1 / s, u.T)

    def fit(self, X, y):
        """
        !!! Must Not Change the Content !!!
        Calculate Mean and Covariance of the model
        :param X: Features, must be 2d array
        :param y: Labels
        """
        nX, dim = X.shape
        classes = np.unique(y)
        priorDict, meanDict = dict(), dict()
        cov = np.zeros((dim, dim))
        # calculate means of each class
        for c in classes:
            XC = np.array([xi for xi, yi in zip(X, y) if yi == c])
            meanDict[c] = np.mean(XC, axis=0)
            priorDict[c] = XC.shape[0] / nX
        # calculate cov
        for xi, yi in zip(X, y):
            diff = (xi - meanDict[yi]).reshape(1, -1)
            cov += np.dot(np.transpose(diff), diff)
        cov /= nX
        self.X, self.y, self.classes = X, y, classes
        self.meanDict, self.cov, self.covInv, self.priorDict =\
            meanDict, cov, self.inv(cov), priorDict
        self.mvndDenominator = np.sqrt(np.power(2 * np.pi, dim) * np.linalg.det(cov))

    def likelihood(self, x, c) -> float:
        """
        !!! Must Not Change the Content !!!
        Calculate likelihood P(Cj|x)
        :param x: a sample
        :param c: class Cj
        :return: P(Cj|x)
        """
        diff = [x - self.meanDict[c]]
        return (np.exp(-1/2 * np.linalg.multi_dot([diff, self.covInv, np.transpose(diff)])) /
                self.mvndDenominator).item()

    def evidence(self, x) -> float:
        """
        !!! Must Not Change the Content !!!
        Calculate Evidence P(x)
        P(x) = sum P(x|Cj) * P(Cj)
        :param x: a sample
        :return: P(x)
        """
        p = 0
        for c in self.classes:
            p += self.likelihood(x, c) * self.prior(c)
        return p

    def prior(self, c) -> float:
        """
        !!! Must Not Change the Content !!!
        Calculate Prior P(Cj)
        :param c: class
        :return: P(Cj)
        """
        return self.priorDict[c]

    def predict(self, X) -> (np.ndarray, np.ndarray):
        """
        !!! Must Not Change the Content !!!
        Predict all samples
        :param X: all samples
        :return: (predicted y, array of utility)
        """
        results = np.apply_along_axis(self.predictSample, axis=1, arr=X)
        y = results[:, 0]
        u = results[:, 1]
        return y, u

    def actualUtility(self, yPredicted, yTrue) -> np.ndarray:
        """
        !!! Must Not Change the Content !!!
        Calculate the actual utility of yPredicted
        :param yPredicted: predicted y
        :param yTrue: true y
        :return: array of the actual utility
        """
        return np.array([self.utilityMat[int(yiPredicted), yiTrue] for yiPredicted, yiTrue in zip(yPredicted, yTrue)])

    def posterior(self, x, c) -> float:
        """
        ToDo: Implement This Function
        Calculate posterior probability P(Cj|x)
        P(Cj|x) = P(x|Cj) * P(Cj) / P(x)
        :param x: a sample
        :param c: class
        :return: P(Cj|x)
        """
        return self.likelihood(x, c) * self.prior(c) / self.evidence(x)

    def utility(self, x, action) -> float:
        """
        ToDo: Implement This Function
        Calculate the utility expectation of the action given x
        :param x: a sample
        :param action: choosing which class
        :return: the utility expectation of the action
        """
        
        # E = Sum[ Uij * P(Cj|x) , over all j]
        EU = 0

        i = action                                  # assume action is ai not [ ai's ...]
        for j, cj in enumerate(self.classes):       # j = Cj
            EU += self.utilityMat[i,j] * self.posterior(x, cj)

        #
        #   i = action
        #   EU = sum([ self.utilityMat[i,j] * self.posterior(x, cj) for j, cj in enumerate(self.classes) ])
        #
        return EU

    def predictSample(self, x) -> (int, float):
        """
        ToDo: Implement This Function
        Predict a sample according to the utility expectation
        :param x: a sample
        :return: (predicted yi, utility expectation of choosing yi)
        """

        EU = np.array([ self.utility(x, ai) for ai in self.classes ])
        ai_star = np.argmax(EU)

        return int(ai_star), EU[ai_star]

@timing
def main():
    """
    !!! Must Not Change the Content Within Main Function !!!
    """
    randomState = 0
    resultCsvPath, resultTxtPath = 'Data/results.csv', 'Data/results.txt'

    with Timer('Data Loaded'):
        X, y, _ = loadDataset()
        XTrain, XTest, yTrain, yTest = \
            train_test_split(X, y, test_size=0.2, random_state=randomState)
        print(f'Training Set Length: {XTrain.shape[0]}\n'
              f'Testing Set Length: {XTest.shape[0]}')
        utilityMat = loadUtilityMat()

    classifier = Classifier(utilityMat)
    with Timer('Trained'):
        classifier.fit(XTrain, yTrain)
    with Timer('Tested'):
        yPredicted, uExpected = classifier.predict(XTest)
    uActual = classifier.actualUtility(yPredicted, yTest)
    uExpectedAve, uActualAve = np.average(uExpected), np.average(uActual)

    with Timer('Results Saved'):
        results = pd.DataFrame()
        results['yPredicted'] = yPredicted
        results['yTrue'] = yTest
        results['uExpected'] = uExpected
        results['uActual'] = uActual
        results.to_csv(resultCsvPath, index=False)

        resultStr = f'{classification_report(yTest, yPredicted, digits=5)}\n' \
                    f'Average of Expected Utility: {uExpectedAve}\n' \
                    f'Average of Actual Utility: {uActualAve}'
        with open(resultTxtPath, 'w') as resultFile:
            resultFile.write(resultStr)
    print(resultStr)


if __name__ == '__main__':
    main(timerPrefix='Total Time Costs: ', timerBeep=True)

