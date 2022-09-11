# stdp-nmnist-tales

This is the repository with the code used to run the experiments of the Independent Studies project by Tales Braig in the BICS Lab under supervision of Dr. Lyes Khacef and Prof. Dr. Martin Oettel.

If any questions or doubts arise, feel free to contact me under tales.braig@gmail.com.

File Overview:

NTIDIGITS.py - Script for handling the NTIDIGITS dataset.



modified_bindsnet.py - modified Spiking net from Julian

modified_bindsnet_tales.py - modified Spiking net adapted by Tales Braig adding delays and recurrencies to be included

modified_bindsnet_tales_delta_code.py - modified Spiking net adapted by Tales Braig adding delays and recurrencies to be included and delta code structure



custom_modules_tales.py - New nodes, connections and learning rules proposed and implemented by Tales Braig, used to build the different networks.



population_eval.py - Definitions of population evaluation metrics.



run_and_eval.py - A script to easily build commands to execute several runs automatically by just running one script.



Auditory_network.py - A script running the training and testing (via single neuron evaluation methods) a network on N-TIDIGITS.



Auditory_network_just_train.py - A script running just the training part and saving the network for further testing and evaluation.

Auditory_network_loading_adding_fixed_just_train.py - A script for loading a pretrained model (with all parameters fixed), building a second layer on top and running just the training part on this, as well as saving the second layer for further testing and evaluation.

Auditory_network_delta_code_just_train.py - A script running just the training part on a delta code based network and saving the network for further testing and evaluation.



Auditory_network_loading_test_and_spikerecord.py - A script for loading and running a network on the testing dataset and saving the time summed spiking output activities for further final evaluation.

Auditory_network_loading_test_and_spikerecord_two_layers.py - A script for loading and running a two-layer network on the testing dataset and saving the time summed spiking output activities for further final evaluation.

Auditory_network_loading_test_and_spikerecord_delta_code.py - A script for loading and running a delta code based network on the testing dataset and saving the time summed spiking output activities for further final evaluation.



Anlaytics based on saved activity profiles (after train and test).ipynb - Notebook to load the saved spiking profiles and perform the SVM, population dot product and single neuron evaluations. (NOTE: this has to go into the correct folder or adapted)
