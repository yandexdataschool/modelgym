from sklearn.ensemble import RandomForestClassifier as rfc
from modelgym.model import Model
from hyperopt import hp, fmin, space_eval, tpe, STATUS_OK
from modelgym.XYCDataset import XYCDataset as xycd
from hyperopt.mongoexp import MongoTrials
from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import cross_val_score


class RFModel(Model):

    def __init__(self, learning_task, compute_counters=False, counters_sort_col=None, holdout_size=0):
        Model.__init__(self, learning_task, 'RandomForestClassifier', 
                       compute_counters, counters_sort_col, holdout_size)
        self.space = {
             'max_depth': hp.choice('max_depth', range(1,20)),
             'max_features': hp.choice('max_features', range(1,5)),
             'n_estimators': hp.choice('n_estimators', range(1,20)),
             'criterion': hp.choice('criterion', ["gini", "entropy"]),
        }

        self.default_params = {
             'max_depth': 1,
             'max_features': 4,
             'n_estimators': 10,
             'criterion': "gini"
        }

        self.default_params = self.preprocess_params(self.default_params)
 

    def preprocess_params(self, params):
        #if self.learning_task == "classification":
        params.update({'verbose': 0})
        params['max_depth'] = int(params['max_depth'])
        return params
    
   
    def convert_to_dataset(self, data, label, cat_cols=None):
        ab=xycd(data, label, cat_cols)
        #print(ab)
        return ab

    
    def fit(self, params, dtrain, dtest, n_estimators):
        def hyperopt_train_test(params):
            X_ = dtrain.X[:]
            if 'normalize' in params:
                if params['normalize'] == 1:
                    X_ = normalize(X_)
                params.pop('normalize')

            if 'scale' in params:
                if params['scale'] == 1:
                    X_ = scale(X_)
                params.pop('scale')
            clf = rfc(**params)
            
            return cross_val_score(clf, dtrain.X, dtrain.y).mean()
        
        space4rf = {
            'max_depth': hp.choice('max_depth', range(1,20)),
            'max_features': hp.choice('max_features', range(1,5)),
            'n_estimators': hp.choice('n_estimators', range(1,20)),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
            'scale': hp.choice('scale', [0, 1]),
            'normalize': hp.choice('normalize', [0, 1])
        }
        
        res=[]
        def f(params):
            acc = hyperopt_train_test(params)
            global best
            best=0
            if acc > best:
                best = acc
                #print('new best:', best, params)
                res.append(best)
            return {'loss': -acc, 'status': STATUS_OK}

        best = fmin(f, space4rf, algo=tpe.suggest, max_evals=300)
        params_opt=space_eval(space4rf, best)
        print(params_opt)
        print(params)
        clf=rfc(**params)
        best=clf.fit(dtrain.X,dtrain.y)
        #print(res) 
        return best, res


    def predict(self, bst, dtest, X_test):
        preds = bst.predict(dtest.X)
        return preds
