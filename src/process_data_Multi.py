import codecs
import numpy as np

class SentimentCorpus:
    
    def __init__(self, num_of_class):
        '''
        prepare dataset
        1) build feature dictionaries
        2) split data into train/dev/test sets 
        '''
        
        self.num_of_class = num_of_class
        train_X, train_y, feat_dict, feat_counts = build_dicts_train(self.num_of_class)
        test_X, test_y = build_dicts_test(feat_dict, self.num_of_class) #for testing set same dictonary
        self.nr_instances_train = train_y.shape[0]
        self.nr_features_train = train_X.shape[1]
        self.nr_instances_test = test_y.shape[0]
        self.nr_features_test = test_X.shape[1]
        self.feat_dict = feat_dict
        self.feat_counts = feat_counts
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        


def build_dicts_train(num_of_class):
    '''
    builds feature dictionaries
    ''' 

    feat_counts = {}
    nr_whole = np.zeros(num_of_class, dtype=int)

    for index in range(num_of_class):
        with codecs.open("data/train_data/train_class_" + str(index + 1) + ".review", 'r') as file:
            for line in file:
                nr_whole[index] += 1
                toks = line.split(" ")
                for feat in toks[0:-1]:
                    name, counts = feat.split(":")
                    if name not in feat_counts:
                        feat_counts[name] = 0
                    feat_counts[name] += int(counts)

    to_remove = []
    for key, value in feat_counts.items():
        if value < 5:
            to_remove.append(key)
    for key in to_remove:
        del feat_counts[key]
        
   
    # map feature to index
    feat_dict = {}
    i = 0
    for key in feat_counts.keys():
        feat_dict[key] = i
        i += 1

    nr_feat = len(feat_counts) 
    nr_instances = sum(nr_whole)
    # nr_instances = nr_win + nr_auto + nr_pol
    X = np.zeros((nr_instances, nr_feat), dtype=float)

    size_y = [np.zeros([nr_whole[0], 1], dtype=int)]

    for index in range(1, num_of_class):
        size_y += [index * np.ones([nr_whole[index], 1], dtype=int)] 

    y = np.vstack(tuple(size_y))
    
    nr_whole = np.zeros(num_of_class, dtype=int)

    total = 0
    
    for index in range(num_of_class):
        with codecs.open("data/train_data/train_class_" + str(index + 1) + ".review", 'r') as file:
            for line in file:
                toks = line.split(" ")
                for feat in toks[0:-1]:
                    name, counts = feat.split(":")
                    if name in feat_dict:
                        X[total, feat_dict[name]] = int(counts)
                nr_whole[index] += 1
                total += 1

       
    # shuffle the order, mix windows,autos and politics examples
    new_order = np.arange(nr_instances)
    np.random.seed(0) # set seed
    np.random.shuffle(new_order)
    X = X[new_order,:]
    y = y[new_order,:]

   
    return X, y, feat_dict, feat_counts

def build_dicts_test(feat_dict, num_of_class):
  
    nr_whole = np.zeros(num_of_class, dtype=int)

    for index in range(num_of_class):
        with codecs.open("data/test_data/test_class_" + str(index + 1) + ".review", 'r') as file:
            for line in file:
                nr_whole[index] += 1


    nr_feat = len(feat_dict)
    nr_instances = sum(nr_whole)
    X = np.zeros((nr_instances, nr_feat), dtype=float)

    size_y = [np.zeros([nr_whole[0], 1], dtype=int)]

    for index in range(1, num_of_class):
        size_y += [index * np.ones([nr_whole[index], 1], dtype=int)] 

    y = np.vstack(tuple(size_y))

    total = 0
    for index in range(num_of_class):
        with codecs.open("data/test_data/test_class_" + str(index + 1) + ".review", 'r') as file:
            for line in file:
                toks = line.split(" ")
                for feat in toks[0:-1]:
                    name, counts = feat.split(":")
                    if name in feat_dict:
                        X[total, feat_dict[name]] = int(counts)
                nr_whole[index] += 1
                total += 1
           
  # shuffle the order, mix windows,autos and politics examples
    new_order = np.arange(nr_instances)
    np.random.seed(0) # set seed
    np.random.shuffle(new_order)
    X = X[new_order,:]
    y = y[new_order,:]

   
    return X, y