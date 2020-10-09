import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np

class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, verbose=verbose, scoring=scoring, refit=refit, return_train_score=False)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):

        df = None
        for k in self.keys:
            self.grid_searches[k].cv_results_['estimator'] = k
            if df is not None :
                df = pd.concat([df, pd.DataFrame(self.grid_searches[k].cv_results_)])
            else:
                df = pd.DataFrame(self.grid_searches[k].cv_results_)
        return df
