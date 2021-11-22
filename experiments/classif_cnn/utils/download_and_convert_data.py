import h5py
import json
import time

import torchvision
import torchvision.transforms as transforms
import tqdm


def _dump_dict_to_hdf5(data_dict: dict, hdf5_file: h5py.File):
    '''Dump dict with expected structure to HDF5 file'''

    hdf5_file.create_dataset('users', data=data_dict['users'])
    hdf5_file.create_dataset('num_samples', data=data_dict['num_samples'])

    # Store actual data in groups
    user_data_group = hdf5_file.create_group('user_data')
    for user, user_data in tqdm.tqdm(data_dict['user_data'].items()):
        user_subgroup = user_data_group.create_group(user)
        user_subgroup.create_dataset('x', data=user_data) 

    user_data_label_group = hdf5_file.create_group('user_data_label')
    for user, user_data_label in tqdm.tqdm(data_dict['user_data_label'].items()):
        user_data_label_group.create_dataset(user, data=user_data_label) 

def _process_and_save_to_disk(dataset, n_users, file_format, output):
    '''Process a Torchvision dataset to expected format and save to disk'''

    # Split training data equally among all users
    total_samples = len(dataset)
    samples_per_user = total_samples // n_users
    assert total_samples % n_users == 0

    # Function for getting a given user's data indices
    user_idxs = lambda user_id: slice(user_id * samples_per_user, (user_id + 1) * samples_per_user)

    # Convert training data to expected format
    print('Converting data to expected format...')
    start_time = time.time()

    data_dict = {  # the data is expected to have this format
        'users' : [f'{user_id:04d}' for user_id in range(n_users)],
        'num_samples' : 10000 * [samples_per_user],
        'user_data' : {f'{user_id:04d}': dataset.data[user_idxs(user_id)].tolist() for user_id in range(n_users)},
        'user_data_label': {f'{user_id:04d}': dataset.targets[user_idxs(user_id)] for user_id in range(n_users)},
    }

    print(f'Finished converting data in {time.time() - start_time:.2f}s.')

    # Save training data to disk
    print('Saving data to disk...')
    start_time = time.time()

    if file_format == 'json':
        with open(output + '.json', 'w') as json_file:
            json.dump(data_dict, json_file)
    elif file_format == 'hdf5':
        with h5py.File(output + '.hdf5', 'w') as hdf5_file:
            _dump_dict_to_hdf5(data_dict=data_dict, hdf5_file=hdf5_file)
    else:
        raise ValueError('unknown format.')

    print(f'Finished saving data in {time.time() - start_time:.2f}s.')


# Get training and testing data from torchvision
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)

print('Processing training set...')
_process_and_save_to_disk(trainset, n_users=1000, file_format='hdf5', output='./data/train_data')

print('Processing test set...')
_process_and_save_to_disk(testset, n_users=200, file_format='hdf5', output='./data/test_data')