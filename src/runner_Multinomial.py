from process_data_Multi import SentimentCorpus
from algorithm import MultinomialNaiveBayes
import numpy as np

if __name__ == '__main__':
    dataset = SentimentCorpus(2)
    nb = MultinomialNaiveBayes()

    params = nb.fit(dataset.train_X, dataset.train_y)
    #loglikelihood of each word per class  row= no of words columns =classes
    predict_train = nb.predict(dataset.train_X, params)
    #predict_train has the output of test doc belonging to which class
    eval_train = nb.evaluate(predict_train, dataset.train_y)

    predict_test = nb.predict(dataset.test_X, params)
    
    eval_test = nb.evaluate(predict_test, dataset.test_y)
    
    print("Accuracy on training set: %f, on testing set: %f" % (eval_train, eval_test))