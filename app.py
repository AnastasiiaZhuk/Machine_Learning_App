import streamlit as st
import pandas as pd
import random

from abc import ABCMeta, abstractmethod
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


@st.cache(persist=True)
def load_data():
    data = pd.read_csv("/home/nastiositi/disk_e/PycharmProjects/Machine_Learning_App/mushrooms.csv")
    labelencoder = LabelEncoder()

    for col in data.columns:
        data[col] = labelencoder.fit_transform(data[col])
    return data


@st.cache(persist=True)
def split(df):
    y = df.type
    x = df.drop(columns=['type'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


class Classifier(metaclass=ABCMeta):
    @abstractmethod
    def classify(self, **kwargs):
        pass


class LogisticClassifier(Classifier):

    def classify(self, C, max_iter, penalty='l2'):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C,
                                   penalty='l2',
                                   max_iter=max_iter)
        data = pd.DataFrame([C, max_iter, penalty], ['C', 'max_iter', 'penalty'])

        return model, data


class RandomFrstClassifier(Classifier):

    def classify(self, n_estimators, max_depth, bootstrap, n_jobs=-1):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       bootstrap=bootstrap,
                                       n_jobs=-1)
        data = pd.DataFrame([n_estimators, max_depth, bootstrap, n_jobs], ['n_estimators', 'max_depth', 'bootstrap', 'n_jobs'])

        return model, data


class SVMClassifier(Classifier):

    def classify(self, C, kernel, gamma):
        st.subheader("Support Vector Machine (SVM) Results")
        model = SVC(C=C,
                    kernel=kernel,
                    gamma=gamma)
        data = pd.DataFrame([C, kernel, gamma], ['C', 'kernel', 'gamma'])

        return model, data


class ClassifierFactory:

    df = load_data()
    class_names = ['edible', 'poisonous']
    x_train, x_test, y_train, y_test = split(df)

    @classmethod
    def plot_metrics(cls, metrics_list, model):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, cls.x_test, cls.y_test, display_labels=cls.class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, cls.x_test, cls.y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, cls.x_test, cls.y_test)
            st.pyplot()

    @classmethod
    def get_classificator(cls, classifier):
        model = None

        if classifier == 'Support Vector Machine (SVM)':
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
            kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
            gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
            model = SVMClassifier().classify(C, kernel, gamma)

        elif classifier == 'Logistic Regression':
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
            max_iter = st.sidebar.slider("Maximum number of iterations", 200, 1000, key='max_iter')
            model = LogisticClassifier().classify(C, max_iter)
            return model

        elif classifier == 'Random Forest':
            n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10,
                                                   key='n_estimators')
            max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
            bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
            model = RandomFrstClassifier().classify(n_estimators, max_depth, bootstrap)
            return model
        return model


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",
                                      ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
    model, data = ClassifierFactory.get_classificator(classifier)

    metrics = st.sidebar.multiselect("What metrics to plot?",
                                     ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    factory = ClassifierFactory()

    if st.sidebar.button("Classify", key='classify1'):

        model.fit(factory.x_train, factory.y_train)
        data.columns = ['Using these Parameters']
        st.table(data)
        accuracy = model.score(factory.x_test, factory.y_test)
        y_pred = model.predict(factory.x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(factory.y_test, y_pred, labels=factory.class_names).round(2))
        st.write("Recall: ", recall_score(factory.y_test, y_pred, labels=factory.class_names).round(2))
        factory.plot_metrics(metrics, model)

    if st.sidebar.checkbox("Show raw data", False):

        st.subheader("Mushroom Data Set (Classification)")
        rand1 = random.randrange(1, 10, 1)
        rand2 = random.randint(1, 8100)
        st.write(factory.df[rand2: rand2 + rand1])
        st.markdown(
            "This [data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) includes descriptions"
            "of hypothetical samples corresponding to 23 species of gilled mushrooms "
            "in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified"
            "as definitely edible, definitely poisonous, "
            "or of unknown edibility and not recommended."
            "This latter class was combined with the poisonous one.")


if __name__ == '__main__':
    main()