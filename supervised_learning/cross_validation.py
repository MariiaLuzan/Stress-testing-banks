import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import TimeSeriesSplit
from supervised_learning.estimate_errors import estimate_errors
from sklearn.model_selection import cross_val_score


class PanelDataSplit(BaseCrossValidator):
    """
    This cross-validator is designed for panel data and relies on scikit-learn's 
    TimeSeriesSplit as its foundation. It determines the row indices for both 
    the training and test datasets within each cross-validation fold. 
    TimeSeriesSplit takes into account the temporal aspects of the data and defines 
    which dates are included in the test and training sets for each fold. 
    The PanelDataSplit class defines the row indices corresponding to dates in the 
    training set and the row indices corresponding to dates in the test set for each fold.
    
    Parameters: 
    test_size (integer) - Number of reporting dates in the test set
    date_axis (Pandas Series) - Column containing dates in the dataset used for cross-validation
    n_splits (integer) - Number of splits for cross-validation
    
    Returns:
    panel_train_index, panel_test_index (Generator) - Generator that yields the row indices 
                                                      for the training and test sets
    """
    
    def __init__(self, test_size, date_axis, n_splits=5):
        self.n_splits = n_splits
        self.test_size = test_size
        self.date_axis = date_axis

    def split(self, X, y=None, groups=None):
        
        report_dates = self.date_axis.unique()
        
        # With sklearn's TimeSeriesSplit, determine which dates should be included in the training set and 
        # which dates should be part of the test set for each fold of cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        
        for i, (train_index, test_index) in enumerate(tscv.split(report_dates)):
            
            # Find the row indices that correspond to the dates in the training set
            panel_train_index = self.date_axis.index[self.date_axis.isin(report_dates[train_index])]
            # Find the row indices that correspond to the dates in the testing set
            panel_test_index = self.date_axis.index[self.date_axis.isin(report_dates[test_index])]

            yield panel_train_index, panel_test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    

    
    
def search_best_model(data_set_train, models, grid_search, y_col, fixed_effects):
    """
    Searches the optimal model (factors and hyperparameter values) via cross-validation
    
    Args:
    data_set_train (Pandas Dataframe): DataFrame that contains the training dataset, 
                                       including both features (X) and the response (y)
    models (Dictionary): Dictionary that specifies models and their respective lists of factors 
                         in the format {model: list of factors}
    grid_search (GridSearchCV): GridSearchCV object that defines the estimator, param_grid, 
                                scoring method, and cross-validator
    y_col (String): Name of the column containing the response variable in the data_set_train dataframe
    fixed_effect (List): Names of columns for fixed effects

    Returns:
    models_results (Pandas Dataframe): Results of the GridSearchCV process for each model from the models dictionary
    best_model_name (String): The name of the best-performing model
    best_score (Float): The highest cross-validation score achieved by the selected best model
    best_model (Pipeline): The best estimator for the chosen optimal model
    """
    
    # Name of best the best model (corresponds to names in the models disctionary)
    best_model_name = None
    # Pipeline for the best model
    best_model = None
    # Cross-validation score for the best model
    best_score = -1000
    
    estimators = []
    
    models_results={}
    
    # Loop over all models in the models dictionary
    for model in models:
        model_factors = models[model]
        model_factors_all = model_factors + fixed_effects
        
        # Define X_train and y_train for the model
        X_train = data_set_train[model_factors_all]
        y_train = data_set_train[y_col]
        
        # Grid Search for the model
        grid_search.fit(X_train, y_train)
        
        models_results[model] = {'Cross-Validation R^2': grid_search.best_score_,
                                 'Cross-Validation R^2 std': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
                                 'Best Hyperparameters': grid_search.best_params_,
                                }
        
        estimators.append(grid_search.best_estimator_)
        
        if best_score < grid_search.best_score_:
            best_model_name = model
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
    
    models_results = pd.DataFrame.from_dict(models_results, orient='index')
    
    return best_model_name, best_score, best_model, models_results, estimators    


def ablation_analysis(estimator, data_set_train,
                      features_sets, y_col,
                      cv, scoring='r2'):
    """
    Returns the results of an ablation analysis by calculating cross-validation scores 
    for different sets of features. Each set of features is obtained by excluding some 
    specific factor or group of factors.
    
    Args:
    estimator (Pipeline) - The model to be tested
    data_set_train (Pandas DataFrame) - DataFrame containing the dataset
    features_sets (Dictionary) - Dictionary containing feature sets for testing in the format: 
                                 {set_name: list of features}
    y_col (String): Name of the column containing the response variable
    cv (Custom cross-validation class) - Cross-validator 
    scoring (String) - Type of cross-validation score

    Returns: 
    ablation_results (Pandas DataFrame) - DataFrame containing cross-validation scores for the specified feature sets
    """
    
    ablation_results={}
    
    y_train = data_set_train[y_col]
    
    for feature_set in features_sets:
        columns = features_sets[feature_set]
        X_train = data_set_train[columns]
        cv_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='r2')
        ablation_results[feature_set] = {'R^2 mean': cv_scores.mean(),
                                         'R^2 standard error of the mean': cv_scores.std() / cv.get_n_splits()**0.5
                                        }
    ablation_results = pd.DataFrame.from_dict(ablation_results, orient='index')
    
    return ablation_results 


def find_Lasso_coef(data_set_train, data_set_test, grid_search, 
                    model_factors, y_col, fixed_effects, 
                    lower_limit = None, upper_limit=None):
    """
    Estimates Lasso Coefficients for a model (factor set)
    
    Args:
    data_set_train (Pandas Dataframe): DataFrame that contains the training dataset, 
                                       including both features (X) and the response (y)
    model_factors (List): List that specifies a set of factors for the model
    grid_search (GridSearchCV): GridSearchCV object that defines the estimator, param_grid, 
                                scoring method, and cross-validator
    y_col (String): Name of the column containing the response variable
    fixed_effect (List): Names of columns for fixed effects
    lower_limit (Float): y_test values lower than this limit are treated as outliers and excluded from the calculation of errors
    upper_limit (Float): y_test values higher than this limit are treated as outliers and excluded from the calculation of errors

    Returns:
    model_result (Pandas Dataframe): Errors computed on the test dataset for the model selected through cross-validation
    model_coef (Pandas Dataframe): Coefficients estimated for the model chosen during cross-validation
    
    """
    
    model_factors_all = model_factors + fixed_effects
        
    # Define X_train and y_train for the model
    X_train = data_set_train[model_factors_all]
    y_train = data_set_train[y_col]
        
    # Grid Search for the model
    grid_search.fit(X_train, y_train)
    
    # Best pipeline
    best_model = grid_search.best_estimator_
    
    # Fit the pipeline on the whole train set
    best_model.fit(X_train, y_train)
    
    # Best model coefficients
    model_coef = pd.DataFrame(data={'factors': model_factors_all, 'coef': best_model[1].coef_})
    model_coef = model_coef.iloc[:len(model_factors)]
    model_coef = model_coef[['factors', 'coef']]
    
    # Results on the test sample
    X_test = data_set_test[model_factors_all]
    y_test = data_set_test[y_col]
    y_pred = best_model.predict(X_test)
    
    model_result = estimate_errors(y_test, y_pred, lower_limit, upper_limit)
        

    
    return model_coef, model_result 


def Lasso_chosen_features(estimator, model_factors, y_col, fixed_effects, data_set_train):
    """
    Fits the pipeline to the entire training sample and provides the coefficients of the Lasso regression from this pipeline
    
    Args:
    estimator (Pipeline) - Chosen pipeline
    model_factors (List) - List containing th model's features 
    y_col (String) - Name of the column containing the response variable
    fixed_effects (List): Names of columns for fixed effects 
    data_set_train (Pandas DataFrame) - DataFrame containing the dataset
    
    Returns:
    model_coef (Pandas Dataframe) - DataFrame containing the coefficients of the Lasso model
    """
    
    model_factors_all = model_factors + fixed_effects
    
    # Define X_train and y_train for the model
    X_train = data_set_train[model_factors_all]
    y_train = data_set_train[y_col]
    
    # Fit the pipeline on the whole train set
    estimator.fit(X_train, y_train)
    
     # Best model coefficients
    model_coef = pd.DataFrame(data={'factors': model_factors_all, 'coef': estimator[1].coef_})
    model_coef = model_coef.iloc[:len(model_factors)]
    model_coef = model_coef[['factors', 'coef']]
    
    return model_coef