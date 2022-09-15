params_Q = dict()
# Neural Network
params_Q['epsilon_decay_linear'] = 1/700 # exploit after 500 games
params_Q['learning_rate'] = 0.0013629
# params_Q['first_layer_size'] = 200    # neurons in the first layer
# params_Q['second_layer_size'] = 20   # neurons in the second layer
# params_Q['third_layer_size'] = 50    # neurons in the third layer
params_Q['episodes'] = 10000
params_Q['memory_size'] = 25000 #too short ???
params_Q['batch_size'] = 3000
params_Q['update_frequency'] = 4
params_Q['net_update_frequency'] = 10000
params_Q['replay_start'] = 500
# Settings
params_Q['weights_path'] = 'weights/weights.h5'
params_Q['load_weights'] = False
params_Q['train'] = True
params_Q["test"] = False
params_Q['plot_score'] = True
# params_Q['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
