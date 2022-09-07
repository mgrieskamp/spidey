params_Q = dict()
# Neural Network
params_Q['epsilon_decay_linear'] = 1/100
params_Q['learning_rate'] = 0.00013629
params_Q['first_layer_size'] = 200    # neurons in the first layer
params_Q['second_layer_size'] = 20   # neurons in the second layer
params_Q['third_layer_size'] = 50    # neurons in the third layer
params_Q['episodes'] = 250
params_Q['memory_size'] = 2500
params_Q['batch_size'] = 1000
# Settings
params_Q['weights_path'] = 'weights/weights.h5'
params_Q['train'] = False
params_Q["test"] = True
params_Q['plot_score'] = True
# params_Q['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
