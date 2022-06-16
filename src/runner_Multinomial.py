from process_data_Multi import SentimentCorpus
from algorithm import MultinomialNaiveBayes
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import time

if __name__ == '__main__':
    dataset = SentimentCorpus(20)

    #------------
    nb = MultinomialNaiveBayes()

    params = nb.fit(dataset.train_X, dataset.train_y)
    #loglikelihood of each word per class  row= no of words columns =classes
    predict_train = nb.predict(dataset.train_X, params)
    #predict_train has the output of test doc belonging to which class
    eval_train = nb.evaluate(predict_train, dataset.train_y)

    predict_test = nb.predict(dataset.test_X, params)
    
    eval_test = nb.evaluate(predict_test, dataset.test_y)
    

    print("My model")
    print("Accuracy on training set: %f, on testing set: %f" % (eval_train, eval_test))

    #MultinomialNB from scikit-learn
    clf = MultinomialNB()

    #training model
    clf.fit(dataset.train_X, dataset.train_y)

    sk_predict_train = clf.predict(dataset.train_X)
    sk_evaluate_train = nb.evaluate(sk_predict_train, dataset.train_y)

    sk_predict_test = clf.predict(dataset.test_X)
    sk_evaluate_test = nb.evaluate(sk_predict_test, dataset.test_y)

    print("SK learn model")
    print("Accuracy on training set: %f, on testing set: %f" % (sk_evaluate_train, sk_evaluate_test))