# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time
import torchvision
import torchvision.transforms as transforms

class CIFAR10:
    def __init__(self) :
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
        self.trainset=_process(trainset, n_users=1000)

        print('Processing test set...')
        self.testset=_process(testset, n_users=200)

def _process(dataset, n_users):
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

    return data_dict

