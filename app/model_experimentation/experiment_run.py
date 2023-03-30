from hyperopt.pyll.base import scope

import os
import model_experimentation.model_utilities as mu
import pandas as pd
import mlflow
import hyperopt
import time
import boto3
import yaml

# Load config
with open(os.path.dirname(os.path.dirname(__file__)) + '/config.yaml', 'r') as file:
	config = yaml.safe_load(file)

MLFLOW_TRACKING_URI = config["MLFLOW_TRACKING_URI"]
AWS_S3_BUCKET = config["AWS_S3_BUCKET"]
AWS_ACCESS_KEY_ID = config["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = config["AWS_SECRET_ACCESS_KEY"]

s3 = boto3.client('s3', aws_access_key_id = AWS_ACCESS_KEY_ID, aws_secret_access_key = AWS_SECRET_ACCESS_KEY)


def run_experiment(model_params_args_dict: dict) -> None:

    """
    Runs an experiment based on input parameters provided through a dictionary.
    
    Parameters:
        model_params_args_dict: a dictionary containing the following keys:
            - current_date (str): the current date in YYYY-MM-DD format
            - name (str): the name of the experiment
            - model_type (str): the type of the model (XGBoost or SVM)
            - model_class (str): the class of the model (ML or algo)
            - ticker (str): the stock ticker
            - target (str): the target feature
            - horizon (int): the forecast horizon
            - window (int): the training window
            - features (list): a list of features
            - max_evals (int): the maximum number of evaluations
            - scores (list): a list of scores
            - parallelism (int): the number of parallel processes to use
            
    Returns:
        None
    """
    
    # Set Parameters
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    CURRENT_DATE =  model_params_args_dict['current_date']
    NAME = model_params_args_dict['name']
    MODEL_TYPE = model_params_args_dict['model_type']
    MODEL_CLASS = model_params_args_dict['model_class']
    STOCK = model_params_args_dict['ticker']
    TARGET = model_params_args_dict['target']
    HORIZON = model_params_args_dict['horizon']
    WINDOW = model_params_args_dict['window']
    FEATURES = model_params_args_dict['features']
    MAX_EVALS = model_params_args_dict['max_evals']
    SCORES = model_params_args_dict['scores']
    PARALLELISM = model_params_args_dict['parallelism']

    if MODEL_CLASS == "ML":
        if MODEL_TYPE == 'xgboost':
            HYPER_PARAMETER_SPACE = {

                'max_depth': scope.int(hyperopt.hp.quniform('max_depth', 
                                                            model_params_args_dict['max_depth']['min'], 
                                                            model_params_args_dict['max_depth']['max'],
                                                            model_params_args_dict['max_depth']['step'])),

                'n_estimators': scope.int(hyperopt.hp.quniform('n_estimators', 
                                                                model_params_args_dict['n_estimators']['min'], 
                                                                model_params_args_dict['n_estimators']['max'],
                                                                model_params_args_dict['n_estimators']['step'])),

                'reg_alpha': hyperopt.hp.uniform('reg_alpha',
                                                model_params_args_dict['reg_alpha']['min'], 
                                                model_params_args_dict['reg_alpha']['max']),

                'max_delta_step' : hyperopt.hp.uniform('max_delta_step',
                                                        model_params_args_dict['max_delta_step']['min'], 
                                                        model_params_args_dict['max_delta_step']['max']),

                'min_split_loss' : hyperopt.hp.uniform('min_split_loss',
                                                    model_params_args_dict['min_split_loss']['min'], 
                                                    model_params_args_dict['min_split_loss']['max']),

                'learning_rate': hyperopt.hp.uniform('learning_rate',
                                                    model_params_args_dict['learning_rate']['min'], 
                                                    model_params_args_dict['learning_rate']['max']),

                'min_child_weight': hyperopt.hp.uniform('min_child_weight', 
                                                        model_params_args_dict['min_child_weight']['min'], 
                                                        model_params_args_dict['min_child_weight']['max']),

                'scale_pos_weight': hyperopt.hp.uniform('scale_pos_weight', 
                                                        model_params_args_dict['scale_pos_weight']['min'], 
                                                        model_params_args_dict['scale_pos_weight']['max'])
            }

        elif MODEL_TYPE == 'svm':
            HYPER_PARAMETER_SPACE = {
                'C': hyperopt.hp.uniform('C',
                                        model_params_args_dict['C']['min'], 
                                        model_params_args_dict['C']['max']),

                'gamma': hyperopt.hp.uniform('gamma',
                                            model_params_args_dict['gamma']['min'], 
                                            model_params_args_dict['gamma']['max']),

                'degree': scope.int(hyperopt.hp.quniform('degree',
                                                        model_params_args_dict['degree']['min'], 
                                                        model_params_args_dict['degree']['max'],
                                                        model_params_args_dict['degree']['step'])),

                'kernel': hyperopt.hp.choice('kernel',model_params_args_dict['kernel']['choice'])
            }

    elif MODEL_CLASS == "algo":
        if MODEL_TYPE == "moving_average":
            HYPER_PARAMETER_SPACE = {
                "short_term_ma": scope.int(hyperopt.hp.quniform('short_term_ma', 
                                                                model_params_args_dict['short_term_ma']['min'], 
                                                                model_params_args_dict['short_term_ma']['max'],
                                                                model_params_args_dict['short_term_ma']['step'])),

                "long_term_ma": scope.int(hyperopt.hp.quniform('long_term_ma', 
                                                                model_params_args_dict['long_term_ma']['min'], 
                                                                model_params_args_dict['long_term_ma']['max'],
                                                                model_params_args_dict['long_term_ma']['step']))
            }


    #try:
    #    assert(all(model in mu.MODEL_ATTRIBUTES[MODEL_CLASS].keys() for model in HYPER_PARAMETER_SPACE[MODEL_CLASS].keys()))
    #except:
    #    raise ValueError("Model Type in hyperparameter space not found in list of supported models")
    

    # NOTE: Please update the experiment names to reflect the convention in the instructions above
    EXPERIMENTAL_INPUTS = {
        f'{MODEL_CLASS}_{CURRENT_DATE}_{STOCK}_{NAME}':
            {
                'OPTIMIZE_METRIC': 'returns'
                }
    }
    
    #try:
    #    assert(all(val['MODEL_TYPE'] in HYPER_PARAMETER_SPACE[MODEL_CLASS].keys() for val in EXPERIMENTAL_INPUTS.values()))
    #except:
    #    raise ValueError("Model Type in EXPERIMENTAL_INPUTS not found in HYPER_PARAMETER_SPACE")
    #try:
    #    assert(all(val['MODEL_TYPE'] in mu.MODEL_ATTRIBUTES.keys() for val in EXPERIMENTAL_INPUTS.values()))
    #except:
    #    raise ValueError("Model Type in EXPERIMENTAL_INPUTS not found in list of supported models")
    
    
    # Run through each experiment
    for EXPERIMENT_NAME, EXPERIMENT in EXPERIMENTAL_INPUTS.items():
        runs = mlflow.search_runs()
        if 'tags.mlflow.runName' in runs.columns and EXPERIMENT_NAME in runs['tags.mlflow.runName'].values:
            continue
    
        # Generate features and labels data from assets assocaited with each experiment
        response = s3.get_object(Bucket = AWS_S3_BUCKET, Key = f"prices/{STOCK}.csv")
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            print(f"Successful S3 get_object response. Status - {status}")
            data = pd.read_csv(response.get("Body"))
        else:
            print(f"Unsuccessful S3 get_object response. Status - {status}")

        # Filter data up to current date
        data = data[data['t'] <= CURRENT_DATE]

        # set up data such that we are predicting the next day's stock price direction
        data[TARGET] = data[TARGET].shift(-1)
        data.dropna(inplace = True)
        #train_X, train_y = data[FEATURES], data[TARGET]

        # Load experimental variables
        fitted_metric = EXPERIMENT['OPTIMIZE_METRIC']   
        #model_type = EXPERIMENT['MODEL_TYPE']

        # Define objective function to optimize the model on
        trials = hyperopt.SparkTrials(parallelism = PARALLELISM)

        train_objective = mu.build_train_objective(train_data = data,
                                                   features = FEATURES,
                                                   target = TARGET,
                                                   scores = SCORES,
                                                   model_type = MODEL_TYPE,
                                                   model_class = MODEL_CLASS,
                                                   horizon = HORIZON,
                                                   window = WINDOW,
                                                   run_name = EXPERIMENT_NAME)
    
        # Run the experiment with objective function defined above
        with mlflow.start_run(run_name = EXPERIMENT_NAME):
        
            start = time.time()
        
            # Log tags for distinguishability
            mlflow.set_tag("target_variable", TARGET)
            mlflow.set_tag("features", FEATURES)
            mlflow.set_tag("model_class", MODEL_CLASS)
            mlflow.set_tag("train_date", CURRENT_DATE)
                
            # Hyperopt optimizer, for further reading:
            # https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf
            hyperopt.fmin(fn = train_objective,
                          space = HYPER_PARAMETER_SPACE,
                          algo = hyperopt.tpe.suggest,
                          max_evals = MAX_EVALS,
                          trials = trials)
        
            # Best set of parameters found from optimizer
            best_params = mu.log_and_return_best_params(fitted_metric = fitted_metric,
                                                        scores = SCORES,
                                                        space = HYPER_PARAMETER_SPACE,
                                                        experiment_name = EXPERIMENT_NAME)
            
            # Fit and log model using based on best parameters and store into the registry
            mu.fit_and_log_model(model_params = best_params,
                                        train_data = data,
                                        features = FEATURES,
                                        target = TARGET,
                                        model_type = MODEL_TYPE,
                                        model_class = MODEL_CLASS,
                                        fitted_metric = fitted_metric)
            
            mlflow.set_tag("run_time", str((time.time() - start)))
 
if __name__ == "__main__":
    run_experiment()