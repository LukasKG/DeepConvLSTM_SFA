from datetime import datetime
from os import makedirs
from sys import maxsize

import data_source as ds
from params import Params

from DeepConvLSTM_new.datasets import SensorDataset
from DeepConvLSTM_new.DeepConvLSTM_py3 import DeepConvLSTM
from DeepConvLSTM_new.DeepConvLSTM_py3 import model_train
from DeepConvLSTM_new.DeepConvLSTM_py3 import model_eval

def run_DeepConv(P,conv_layers=4,epochs=30,verbose=True):
    config_dataset = {
            "dataset": P.get('dataset'),
            "window": P.get('winsize'),
            "stride": P.get('jumpsize'),
            "path": P.get('output_path'),
        }

    dataset = SensorDataset(**config_dataset, data=P.F[0], prefix="train")
    #dataset.get_info()

    dataset_val = SensorDataset(**config_dataset, data=P.F[1], prefix="val")
    #dataset_val.get_info()

    dataset_test = SensorDataset(**config_dataset, data=P.F[2], prefix="test")
    #dataset_test.get_info()


    n_classes = len(P.get('labels'))
    n_channels = dataset.n_channels


    deepconv = DeepConvLSTM(n_channels=n_channels, n_classes=n_classes, conv_layers=conv_layers, dataset=P.get('dataset')).cuda()



    # Define train config options
    config_train = {'batch_size': 256,
                    'optimizer': 'Adam',
                    'lr': 1e-3,
                    'lr_step': 10,
                    'lr_decay': 0.9,
                    'init_weights': 'orthogonal',
                    'epochs': P.get('epochs'),
                    'print_freq': 100
                   }

    model_train(deepconv, dataset, dataset_val, config_train, verbose=verbose)


    test_config = {'batch_size': 256,
                  'train_mode': False,
                  'dataset': P.get('dataset'),
                  'num_batches_eval': 212}

    acc_test, fm_test, fw_test, elapsed = model_eval(deepconv, dataset_test, test_config, return_results=True)
    P.log(f"[-] Test acc: {100 * acc_test:.2f}(%)\tfm: {100 * fm_test:.2f}(%)\tfw: {100 * fw_test:.2f}(%)")
    return acc_test, fm_test, fw_test, elapsed

def test_param(param_args,param_name='No Name',param_list=None,degree_list=[1,2],init_print=True):
    P = Params(**param_args,init_print=init_print)
    
    save_path = f"results/{P.get('name')}_R{P.get('run')}"
    results = np.zeros((3,len(degree_list),len(param_list)))

    for i,degree in enumerate(degree_list):
        for j,param_val in enumerate(param_list):
            param_args['degree'] = degree
            param_args[param_name] = param_val
            P_run = Params(**param_args)
            P.log(f"Run {degree=} P[{param_name}]={P_run.get(param_name)} (Run {P_run.get('run')+1})")
            acc_test, fm_test, fw_test, elapsed = run_DeepConv(P_run,conv_layers=P.get('conv_layers'))
            results[0,i,j] = acc_test
            results[1,i,j] = fm_test
            results[2,i,j] = fw_test     
    
    P.log(f"Baseline")
    param_args['SFA'] = False
        
    base_results = np.zeros((3))
    P_run = Params(**param_args)
    acc_base, fm_base, fw_base, elapsed_base = run_DeepConv(P_run,conv_layers=P.get('conv_layers'))
    base_results[0] = acc_base
    base_results[1] = fm_base
    base_results[2] = fw_base
        
    return results, base_results

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--short', action='store_true',dest='short',help='Activate test mode.')
    parser.add_argument('-r','--runs', type=int, help='Number of runs.',default=maxsize,dest='runs')
    parser.add_argument('-e','--epochs', type=int, help='Number of epochs.',default=300,dest='epochs')
    args = parser.parse_args()
    
    param_args = {}

    param_args['name'] = 'run'
    param_args['verbose'] = False

    param_args['dataset'] = 'SHL_ext'
    param_args['dataset'] = 'User1'
    param_args['dataset'] = 'SHL'
    param_args['dataset'] = 'User1s'

    param_args['labels'] = [1,2,3,4,5,6,7,8]
    
    param_args['label_idx'] = True

    param_args['noise'] = 0.0

    param_args['channels'] = 'acc_mag'
    #param_args['channels'] = 'acc'
    #param_args['channels'] = 'both_mag'

    param_args['winsize'] = 500

    param_args['jumpsize'] = 500
    param_args['window_channels'] = True

    ### SFA Params ###
    param_args['SFA'] = True
    param_args['save_SFA'] = True
    param_args['load_SFA'] = True
    
    param_args['past_samples'] = 1
    param_args['training_samples'] = 100
    param_args['degree'] = 1
    param_args['iterval'] = 20
    param_args['whitening_dim'] = 5
    param_args['output_dim'] = 2
    
    if args.short:
        param_args['name'] = 'Short_Run'
        param_args['dataset'] = 'Short'
        param_args['labels'] = [1,2,4]
        param_args['save_SFA'] = False
        param_args['load_SFA'] = False


    runs = args.runs
    epochs = args.epochs
    past_sample_list = np.array([3,5,8,15,40,70,100])
    degrees = np.array(range(1,3))
    conv_layer_list = [1,2,3,4]
    
    if args.short:
        runs = 2
        epochs = 5
        past_sample_list = np.array([1,2])
        degrees = np.array([1])
        conv_layer_list = [1]
        
        
    param_args['runs'] = runs
    param_args['epochs'] = epochs
    param_args['conv_layers'] = 4
    
    P = Params(**param_args)
    
    makedirs('results/', exist_ok=True)
    result_mat = ds.load_file("results/result_mat.npy")
    if result_mat is None:
        result_mat = np.zeros((0,len(degrees),len(conv_layer_list),len(past_sample_list)+1))
         
    run = result_mat.shape[0]
    while run < param_args['runs']:
        param_args['run'] = run
        new_mat = np.zeros((1,len(degrees),len(conv_layer_list),len(past_sample_list)+1))
        result_mat = np.concatenate((result_mat,new_mat), axis=0)
        for c,conv_layers in enumerate(conv_layer_list):
            param_args['conv_layers'] = conv_layers
            param_args['name'] = f"E{epochs}_CL{conv_layers}"
            P = Params(**param_args,init_print=True)
            
            results, base_results = test_param(
                param_args=param_args,
                param_name='past_samples',
                param_list=past_sample_list,
                degree_list=degrees,
                init_print=False)

            for j,_ in enumerate(degrees):
                result_mat[run,j,c,0] = base_results[2]
                result_mat[run,j,c,1:] = results[2,j]

            results_mean = np.mean(result_mat[:,:,c,1:],axis=0)
   
            base_results_mean = np.mean(result_mat[:,0,c,0],axis=0)
            base_vector = np.zeros((past_sample_list.shape[0]))
            base_vector.fill(base_results_mean)

            for scale in ['linear','log']:
                for i,degree in enumerate(degrees):
                    fig, ax = plt.subplots()
                    ax.set_title(f"F1 weighted {degree=}")
                    ax.set_ylabel('%')
                    ax.set_ylim([0, 1])
                    ax.set_xlabel('Past Samples')
                    ax.set_xlim([past_sample_list[0]-1, past_sample_list[-1]+1])
                    ax.set_xscale(scale)

                    ax.plot(past_sample_list,results_mean[i], label='Performance')
                    ax.plot(past_sample_list,base_vector, label='Baseline')
                    ax.grid()
                    ax.legend()

                    ds.save_fig(P,f"f1_weighted_{degree=}_{scale=}",fig,close=True)
                
        P.log(f"Save results {run=} {result_mat.shape=}")
        ds.save_file("results/result_mat.npy", result_mat)
        run+=1
    
    # Create Figures
    for c,conv_layers in enumerate(conv_layer_list):
        param_args['conv_layers'] = conv_layers
        param_args['name'] = f"E{epochs}_CL{conv_layers}"
        P = Params(**param_args)
            
        results_mean = np.mean(result_mat[:,:,c,1:],axis=0)

        base_results_mean = np.mean(result_mat[:,0,c,0],axis=0)
        base_vector = np.zeros((past_sample_list.shape[0]))
        base_vector.fill(base_results_mean)

        for scale in ['linear','log']:
            for i,degree in enumerate(degrees):
                fig, ax = plt.subplots()
                ax.set_title(f"F1 weighted {degree=}")
                ax.set_ylabel('%')
                ax.set_ylim([0, 1])
                ax.set_xlabel('Past Samples')
                ax.set_xlim([past_sample_list[0]-1, past_sample_list[-1]+1])
                ax.set_xscale(scale)

                ax.plot(past_sample_list,results_mean[i], label='Performance')
                ax.plot(past_sample_list,base_vector, label='Baseline')
                ax.grid()
                ax.legend()

                ds.save_fig(P,f"f1_weighted_{degree=}_{scale=}",fig,close=True)
    
    print(f"{np.min(result_mat)=}")
    mean_results = np.mean(result_mat,axis=0)
    print(f"{mean_results.shape}")
    for d,degree in enumerate(degrees):  
        df = pd.DataFrame(mean_results[d]*100)
        df=df.round(1)
        df.columns = [0]+list(past_sample_list)
        df.index = conv_layer_list
        df.to_excel(f'E{epochs}_Deg{degree}.xlsx')