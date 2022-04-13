import torch

import torch.utils.data as utils


def get_test(get_data, batch_size, variance, k_value, device):
    x_test, y_test = get_data(batch_size, var=variance)
    x_test, y_test = torch.from_numpy(x_test).float().to(
        device), torch.from_numpy(y_test).long().to(device)
    return x_test, y_test


def get_dataset(get_data, batch_size, npts, variance, k_value):
    samples, labels = get_data(npts, var=variance)
    tensor_samples = torch.stack([torch.Tensor(x) for x in samples])
    tensor_labels = torch.stack([torch.tensor(x) for x in labels])
    dataset = utils.TensorDataset(tensor_samples, tensor_labels)
    train_loader = utils.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=None,
                                    drop_last=True)
    return train_loader
