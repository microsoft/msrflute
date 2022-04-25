# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# '''
# In this file we define the  schema for the configuration 
# files that will be pass it to an instance of the Validator 
# in e2e_trainer.py 
# '''

{
    'model_config':{
            'required': True,
            'type': 'dict',
            'allow_unknown': True,
            'schema': {
                'model_type': {'required': True, 'type':'string'},
                'model_folder': {'required': True, 'type':'string'},
                'BERT':{
                    'required':False,
                    'type': 'dict',
                    'allow_unknown': True,
                    'schema':{
                        'loader_type': {'required': False, 'type': 'string'},
                        'model': {
                            'required': True,
                            'type': 'dict',
                            'allow_unknown': True,
                            'schema': {
                                'model_name_or_path': {'required': False, 'type':'string'},
                                'model_name': {'required': True, 'type':'string'},
                                'process_line_by_line': {'required': True, 'type':'boolean'},
                            }
                        }
                    }
                },
            }
    },

    'dp_config':{
            'required': True,
            'type': 'dict',
            'allow_unknown': True,
            'schema': {
                'enable_local_dp': {'required': True, 'type':'boolean'},
                'enable_global_dp': {'required': False, 'type':'boolean'},
                'eps': {'required': False, 'type':'float'},
                'delta': {'required': False, 'type':'float'},
                'global_sigma': {'required': False, 'type':'float'},
                'max_grad': {'required': False, 'type':'float'},
                'max_weight': {'required': False, 'type':'float'},
                'weight_scaler': {'required': False, 'type':'float'},
                'min_weight': {'required': False, 'type':'float'},
                }
    },

    'privacy_metrics_config':{
            'required': True,
            'type': 'dict',
            'allow_unknown': True,
            'schema': {
                'apply_metrics': {'required': True, 'type':'boolean'},
                'apply_indices_extraction': {'required': False, 'type':'boolean'},
                'allowed_word_rank': {'required': False, 'type':'integer'},
                'apply_leakage_metric': {'required': False, 'type':'boolean'},
                'max_leakage': {'required': False, 'type':'float'},
                'adaptive_leakage_threshold': {'required': False, 'type':'float'},
                'is_leakage_weighted': {'required': False, 'type':'boolean'},
                'attacker_optimizer_config': {'required': False, 'type':'dict', 'allow_unknown': True},
                }
    },

    'strategy':{
        'required': True,
        'type': 'string'
    },

    'server_config':{
            'required': True,
            'type': 'dict',
            'allow_unknown': True,
            'schema': {
                'wantRL': {'required': True, 'type':'boolean', 'allow_unknown': True},
                'RL': {'required': False, 'type':'dict'},
                'resume_from_checkpoint': {'required': True, 'type':'boolean'},
                'do_profiling': {'required': True, 'type':'boolean'},
                'optimizer_config': {
                    'required': True, 
                    'type':'dict',
                    'allow_unknown': True,
                    'schema': {
                        'type': {'required': True, 'type':'string', 'allowed':['sgd', 'adam','adamax', 'lars', 'LarsSGD', 'lamb', 'adamW']},
                        'lr': {'required': True, 'type':'float'},
                        'weight_decay': {'required': False, 'type':'float'},
                    }
                },
                'annealing_config': {
                    'required': True, 
                    'type':'dict',
                    'allow_unknown': True,
                    'schema': {
                        'type': {'required': True, 'type':'string'},
                        'step_interval': {'required': True, 'type':'string'},
                        'gamma': {'required': True, 'type':'float'},
                        'step_size': {'required': True, 'type':'integer'},
                    }
                },
                'val_freq': {'required': False, 'type':'integer', 'default': 1},
                'rec_freq': {'required': False, 'type':'integer', 'default': 8},
                'initial_val': {'required': False, 'type':'boolean', 'default': True},
                'initial_rec': {'required': False, 'type':'boolean', 'default': False},
                'max_iteration': {'required': False, 'type':'integer', 'default': 10000},
                'num_clients_per_iteration': {'required': False, 'type':'integer', 'default': 1},
                'data_config': {
                    'required': True, 
                    'type':'dict',
                    'allow_unknown': True,
                    'keysrules':{'forbidden':['num_clients']},
                    'schema': {
                        'val': {
                            'required': True, 
                            'type':'dict',
                            'allow_unknown': True,
                            'schema': {
                                'batch_size': {'required': False, 'type':'integer', 'default': 40},
                                'loader_type': {'required': False, 'type':'string', 'default':'text'},
                                'val_data': {'required': True, 'type':'string'},
                                'tokenizer_type': {'required': False, 'type':'string'},
                                'prepend_datapath': {'required': False, 'type':'boolean', 'default': False},
                                'vocab_dict': {'required': False, 'type':'string'},
                                'pin_memory': {'required': False, 'type':'boolean', 'default': True},
                                'num_workers': {'required': False, 'type':'integer', 'default': 1},
                                'num_frames': {'required': False, 'type':'integer', 'default': 0},
                                'max_batch_size': {'required': False, 'type':'integer', 'default': 0},
                                'max_num_words': {'required': False, 'type':'integer'},
                                'max_grad_norm': {'required': False, 'type':'float', 'default': 5.0 },
                                'unsorted_batch': {'required': False, 'type':'boolean', 'default': False},
                                'cache_dir': {'required': False, 'type':'string'},
                            },
                        },
                        'test': {
                            'required': True, 
                            'type':'dict',
                            'allow_unknown': True,
                            'schema': {
                                'batch_size': {'required': False, 'type':'integer', 'default': 40},
                                'loader_type': {'required': False, 'type':'string', 'default':'text'},
                                'test_data': {'required': True, 'type':'string'},
                                'tokenizer_type': {'required': False, 'type':'string'},
                                'prepend_datapath': {'required': False, 'type':'boolean', 'default': False},
                                'vocab_dict': {'required': False, 'type':'string'},
                                'pin_memory': {'required': False, 'type':'boolean', 'default': True},
                                'num_workers': {'required': False, 'type':'integer', 'default': 1},
                                'num_frames': {'required': False, 'type':'integer', 'default': 0},
                                'max_batch_size': {'required': False, 'type':'integer', 'default': 0},
                                'max_num_words': {'required': False, 'type':'integer'},
                                'max_grad_norm': {'required': False, 'type':'float', 'default': 5.0 },
                                'unsorted_batch': {'required': False, 'type':'boolean', 'default': False},
                                'cache_dir': {'required': False, 'type':'string'},
                            },
                        },
                        'train': {
                            'required': False, 
                            'type':'dict',
                            'allow_unknown': True,
                            'schema': {
                                'batch_size': {'required': False, 'type':'integer', 'default': 40},
                                'loader_type': {'required': False, 'type':'string', 'default':'text'},
                                'train_data': {'required': True, 'type':'string'},
                                'train_data_server': {'required': False, 'type':'string'},
                                'desired_max_samples': {'required': False, 'type':'integer'},
                                'tokenizer_type': {'required': False, 'type':'string'},
                                'prepend_datapath': {'required': False, 'type':'boolean', 'default': False},
                                'vocab_dict': {'required': False, 'type':'string'},
                                'pin_memory': {'required': False, 'type':'boolean', 'default': True},
                                'num_workers': {'required': False, 'type':'integer', 'default': 1},
                                'num_frames': {'required': False, 'type':'integer', 'default': 0},
                                'max_batch_size': {'required': False, 'type':'integer', 'default': 0},
                                'max_num_words': {'required': False, 'type':'integer'},
                                'max_grad_norm': {'required': False, 'type':'float', 'default': 5.0 },
                                'unsorted_batch': {'required': False, 'type':'boolean', 'default': False},
                                'cache_dir': {'required': False, 'type':'string'},
                            }
                        },
                    }
                },
                'type': {
                    'required': False, 
                    'type':'string',
                    'allowed':['model_optimization', 'personalization'],
                    'default': 'model_optimization'
                },
                'aggregate_median': {'required': False, 'type':'string'},
                'initial_lr_client': {'required': True, 'type':'float'},
                'lr_decay_factor': {'required': True, 'type':'float'},
                'weight_train_loss': {'required': True, 'type':'string'},
                'best_model_criterion': {'required': False, 'type':'string', 'default':'loss'},
                'fall_back_to_best_model': {'required': False, 'type':'boolean', 'default': False},
                'softmax_beta': {'required': True, 'type':'float'},
                'server_replay_config': {
                    'required': False, 
                    'type':'dict',
                    'schema':{
                        'server_iterations': {'required': True, 'type':'integer', 'dependencies': 'data_config.train.train_data_server'},
                        'optimizer_config': {
                            'required': True, 
                            'type':'dict',
                            'dependencies': 'data_config.train.train_data_server',
                            'allow_unknown': True,
                            'schema': {
                                'type': {'required': True, 'type':'string', 'allowed':['sgd', 'adam','adamax', 'lars', 'LarsSGD', 'lamb', 'adamW']},
                                'lr': {'required': True, 'type':'float'},
                                'weight_decay': {'required': False, 'type':'float'},
                                'amsgrad': {'required': False, 'type':'boolean'},
                            }
                        },
                    }
                },
                'nbest_task_scheduler': {
                    'required': False, 
                    'type':'dict',
                    'schema':{
                        'num_tasks': {'required': True, 'type':'integer'}, 
                        'iteration_per_task': {'required': True, 'type':'integer'},
                    }
                },
            }
    },

    'client_config':{
        'required': True,
        'type': 'dict',
        'allow_unknown': True,
        'schema': {
            'meta_learning': {'required': False, 'type':'string'},
            'stats_on_smooth_grad': {'required': False, 'type':'boolean'},
            'ignore_subtask': {'required': True, 'type':'boolean'},
            'num_skips_threshold': {'required': False, 'type':'integer'},
            'copying_train_data': {'required': False, 'type':'boolean'},
            'do_profiling': {'required': True, 'type':'boolean'},
            'data_config': {
                'required': True, 
                'type':'dict',
                'allow_unknown': True,
                'keysrules':{'forbidden':['num_clients']},
                'schema': {
                    'train': {
                        'required': True, 
                        'type':'dict',
                        'allow_unknown': True,
                        'schema': {
                            'batch_size': {'required': False, 'type':'integer', 'default': 40},
                            'loader_type': {'required': False, 'type':'string', 'default':'text'},
                            'list_of_train_data': {'required': True, 'type':'string'},
                            'tokenizer_type': {'required': False, 'type':'string'},
                            'prepend_datapath': {'required': False, 'type':'boolean', 'default': False},
                            'vocab_dict': {'required': False, 'type':'string'},
                            'pin_memory': {'required': False, 'type':'boolean', 'default': True},
                            'num_workers': {'required': False, 'type':'integer', 'default': 1},
                            'num_frames': {'required': False, 'type':'integer', 'default': 0},
                            'max_batch_size': {'required': False, 'type':'integer', 'default': 0},
                            'max_num_words': {'required': False, 'type':'integer'},
                            'max_grad_norm': {'required': False, 'type':'float', 'default': 5.0 },
                            'unsorted_batch': {'required': False, 'type':'boolean', 'default': False},
                        }
                    },
                }
            },
            'type': {
                'required': False, 
                'type':'string',
                'allowed':['optimization', 'gradient_computation'],
                'default': 'gradient_computation',
            },
            'meta_optimizer_config': {
                'required': False, 
                'type':'dict',
                'allow_unknown': True,
                'schema': {
                    'type': {'required': True, 'type':'string', 'allowed':['sgd', 'adam','adamax', 'lars', 'LarsSGD', 'lamb', 'adamW']},
                    'lr': {'required': True, 'type':'float'},
                }
            },
            'optimizer_config': {
                'required': True, 
                'type':'dict',
                'allow_unknown': True,
                'schema': {
                    'type': {'required': True, 'type':'string', 'allowed':['sgd', 'adam','adamax', 'lars', 'LarsSGD', 'lamb', 'adamW']},
                    'lr': {'required': False, 'type':'float'},
                    'weight_decay': {'required': False, 'type':'float'},
                }
            },
            'annealing_config': {
                'required': False, 
                'type':'dict',
                'allow_unknown': True,
                'schema': {
                    'type': {'required': True, 'type':'string'},
                    'step_interval': {'required': True, 'type':'string'},
                    'gamma': {'required': False, 'type':'float'},
                    'step_size': {'required': False, 'type':'integer'},
                }
            },
            'ss_config': {'required': False, 'type':'dict', 'allow_unknown': True},
        }
    },
}