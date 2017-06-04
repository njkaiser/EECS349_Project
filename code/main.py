import tensorflow as tf
import model

config_names = ['conv1_kernel_size', 'conv2_kernel_size', 'conv1_num_filters', 'conv2_num_filters', 'dropout_rate']
config_list = []
conv1_kernel_size = [[3, 3],[5, 5], [7, 7]]
conv2_kernel_size = [[3, 3],[5, 5], [7, 7]]
conv1_num_filters = [16, 32]
conv2_num_filters = [32, 64]
dropout_rates     = [0.1, 0.2, 0.3]
for c1ks in conv1_kernel_size:
    for c2ks in conv2_kernel_size:
        for c1nf in conv1_num_filters:
            for c2nf in conv2_num_filters:
                for dr in dropout_rates:
                    config_list.append([c1ks, c2ks, c1nf, c2nf, dr])

# for iteration in range(63,len(config_list)):
#      config = config_list[iteration]
#      model.train_model_loop(iteration, config, config_names, "new_experiment_output.txt")

model.train_model("28_new_architecture_1","2 conv, 2 pool","new_architectures.txt")
