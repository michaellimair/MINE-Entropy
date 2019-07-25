import numpy as np
random_seed = 0
np.random.seed(seed=random_seed)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os
# from model import Mine, LinearReg, Kraskov
from joblib import Parallel, delayed
from . import settings
from tqdm import tqdm
from .util.google_drive_util import GoogleDrive

def saveResultsFig(results_dict, experiment_path=""):
    """
    
    Arguments:
    # results_dict example: 
    # {
    #     'Ground Truth': {
    #         'Gaussian': [(0, 0), (0.2, 0.5), ..., (1,1)],  # [(rho, MI), (rho2, MI_2), ...]
    #         'Bimodal': [(0, 0), (0.2, 0.5), ..., (1,1)]
    #     }, 
    #     'Linear Regression': {
    #         'Gaussian': [(0, 0), (0.2, 0.5), ..., (1,1)],
    #         'Bimodal': [(0, 0), (0.2, 0.5), ..., (1,1)]
    #     }, 
    #     ...
    # }
    """
    # initialize ground truth color
    settings.model['Ground Truth'] = {'color': 'red'}
    
    n_datasets = settings.n_datasets
    # n_columns = settings.n_columns + 1  # 0 to N_Column for visualizing the data, last column for the MI estimate plot

    fig, axes = plt.subplots(nrows=n_datasets, ncols=1, figsize=(12,8))

    for _, (model_name, dataset_results) in enumerate(results_dict.items()):
        for row_id, (dataset_name, results) in enumerate(dataset_results.items()):
            color = settings.model[model_name]['color']
            xs = [x for x, y in results]
            ys = [y for x, y in results]
            if n_datasets > 1:
                axe = axes[row_id]
            else:
                axe = axes
            axe.scatter(xs, ys, edgecolors=color, facecolors='none', label=model_name)
            axe.set_xlabel(settings.data[dataset_name]['varying_param_name'])
            axe.set_ylabel('MI')
            axe.set_title(dataset_name)
            axe.legend()
    figName = os.path.join(experiment_path, "MI")
    fig.savefig(figName, bbox_inches='tight')
    plt.close()

def get_estimation(model_name, model, data_model, data_name, varying_param_name, varying_param_value, experiment_path, pop, batch, rootID, googleDrive):
    """
    Returns: mi estimate (float)
    """

    # results = dict()
    data_model.sample_size = pop

    # data_train = data_model.data
    # data_test = data_model.data
    # if data_train.shape[1]%2 == 1 or data_test.shape[1]%2 == 1:
    #     raise ValueError("dim of data should be even")
    # X_train = data_train[:,0:data_train.shape[1]//2]
    # Y_train = data_train[:,-data_train.shape[1]//2:]
    # X_test = data_test[:,0:data_test.shape[1]//2]
    # Y_test = data_test[:,-data_test.shape[1]//2:]
    ground_truth = data_model.ground_truth

    pathname = "pop={}_batch={}_{}_{}={}_model={}".format(pop, batch, data_name, varying_param_name, varying_param_value,model_name)
    prefix_name_loop = os.path.join(experiment_path, pathname)
    if not os.path.exists(prefix_name_loop):
        os.makedirs(prefix_name_loop, exist_ok=True)
        if googleDrive:
            prefixID = googleDrive.createFolder(pathname, rootID)
    
    # if X_train.shape[1]==1:
    #     #Plot Ground Truth MI
    #     fig, ax = plt.subplots(figsize=(15, 15))
    #     Xmax = max(X_train)
    #     Xmin = min(X_train)
    #     Ymax = max(Y_train)
    #     Ymin = min(Y_train)
    #     x = np.linspace(Xmin, Xmax, 300)
    #     y = np.linspace(Ymin, Ymax, 300)
    #     xs, ys = np.meshgrid(x,y)
    #     ax, c = data_model.plot_i(ax, xs, ys)
    #     fig.colorbar(c, ax=ax)
    #     ax.set_title("i(X;Y)")
    #     figName = os.path.join(prefix_name_loop, "i_XY")
    #     fig.savefig(figName, bbox_inches='tight')
    #     plt.close()


    # Fit Algorithm
    # For plotting extra figure inside the training
    model['model'].batch_size = batch
    model['model'].model_name = model_name
    model['model'].prefix = prefix_name_loop
    model['model'].googleDrive = googleDrive
    model['model'].prefixID = prefixID
    # model['model'].prefix = os.path.join(prefix_name_loop, model_name)
    # if not os.path.exists(model['model'].prefix):
    #     os.makedirs(model['model'].prefix)
    model['model'].paramName = varying_param_name
    model['model'].paramValue = varying_param_value
    model['model'].ground_truth = ground_truth
    # mi_estimation = model['model'].predict(X_train, Y_train, X_test, Y_test)
    mi_estimation = model['model'].predict(data_model)
    model['model'].load_all_array()
    model['model'].save_figure(suffix="all")


    # Save Results
    # results[model_name] = mi_estimation

    # Ground Truth
    # results['Ground Truth'] = ground_truth

    return mi_estimation, ground_truth, model_name, data_name, varying_param_value

def plot(experiment_path, rootID, googleDrive):
    # Initialize the results dictionary

    # results example: 
    # {
    #     'Ground Truth': {
    #         'Gaussian': [(0, 0), (0.2, 0.5), ..., (1,1)],  # [(rho, MI), (rho2, MI_2), ...]
    #         'Bimodal': [(0, 0), (0.2, 0.5), ..., (1,1)]
    #     }, 
    #     'Linear Regression': {
    #         'Gaussian': [(0, 0), (0.2, 0.5), ..., (1,1)],
    #         'Bimodal': [(0, 0), (0.2, 0.5), ..., (1,1)]
    #     }, 
    #     ...
    # }


    results = dict()
    results['Ground Truth'] = dict()
    for model_name in settings.model.keys():
        results[model_name] = dict()
        for data_name in settings.data.keys():
            results[model_name][data_name] = []
            results['Ground Truth'][data_name] = []

    # for data_name, data in settings.data.items():
    #     for kwargs in data['kwargs']:
    #         varying_param_name = data['varying_param_name']
    #         varying_param_value = kwargs[varying_param_name]
    #         prefix_name_loop = os.path.join(experiment_path, "{}_{}={}/".format(data_name, varying_param_name, varying_param_value))
    #         if not os.path.exists(prefix_name_loop):
    #             os.makedirs(prefix_name_loop)
    
    # # Main Loop
    r = Parallel(n_jobs=settings.cpu)(delayed(get_estimation)(model_name, 
                                                              model, 
                                                              data['model'](**kwargs), 
                                                              data_name, 
                                                              data['varying_param_name'], 
                                                              kwargs[data['varying_param_name']], 
                                                              experiment_path,
                                                              pop_,
                                                              batch_,
                                                              rootID,
                                                              googleDrive) 
                                                                    for pop_, batch_ in tqdm(settings.pop_batch)
                                                                    for model_name, model in tqdm(settings.model.items())
                                                                    for data_name, data in tqdm(settings.data.items())
                                                                    for kwargs in tqdm(data['kwargs'])
                                                                    )
    for (mi_estimate, ground_truth, model_name, data_name, varying_param_value) in r:
        results[model_name][data_name].append((varying_param_value, mi_estimate))
        results['Ground Truth'][data_name].append((varying_param_value, ground_truth))
    # Plot and save
    saveResultsFig(results, experiment_path=experiment_path)

    return 0

def run_experiment():
    # prompt
    experiment_name = input('Please enter the experiment name: ')
    experiment_path = os.path.join(settings.output_path, experiment_name)
    googleDrive = GoogleDrive()
    googleDrive.connect()
    while True:
        if os.path.exists(experiment_path):
            rootID = googleDrive.searchFolder(experiment_name)
            if not rootID:
                raise ValueError("folder {} not found in googledrive".format(experiment_name))
            # experiment_name = input('experiment - \"{}\" already exists! Please re-enter the experiment name: '.format(experiment_name))
            # experiment_path = os.path.join(settings.output_path, experiment_name)
            break
        else:
            os.makedirs(experiment_path)
            rootID = googleDrive.createFolder(experiment_name)
            print('Output will be saved into {}'.format(experiment_path))
            # save the settings
            from shutil import copyfile
            mmi_dir_path = os.path.dirname(os.path.abspath(__file__))
            settings_path = os.path.join(mmi_dir_path, 'settings.py')
            copyfile(settings_path, os.path.join(experiment_path, 'settings.py'))
            googleDrive.uploadFile(settings_path, 'settings.py', rootID)
            break     
    plot(experiment_path, rootID, googleDrive)

if __name__ == "__main__":
    run_experiment()
    # run_experiment_batch_pop_ir()
