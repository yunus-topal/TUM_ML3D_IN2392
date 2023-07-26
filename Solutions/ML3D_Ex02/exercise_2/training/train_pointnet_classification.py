from pathlib import Path

import torch

from exercise_2.data.shapenet import ShapeNetPoints
from exercise_2.model.pointnet import PointNetClassification


def train(model, trainloader, valloader, device, config):

    # TODO Declare loss and move to specified device
    loss_criterion = torch.nn.CrossEntropyLoss()

    # TODO Declare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    # TODO Implement the training loop. It looks very much the same as in the previous exercise part, except that you are now using points instead of voxel grids

    for epoch in range(config['max_epochs']):
        
        total_running_loss_train = 0.0
        total_count = 0

        best_accuracy = 0.0

        for train_batch in trainloader:
            ShapeNetPoints.move_batch_to_device(train_batch, device)

            optimizer.zero_grad()

            # TODO Forward pass
            output = model(train_batch['points'])

            # TODO Compute loss
            loss = loss_criterion(output, train_batch['label'])

            # TODO Backward pass
            loss.backward()

            # TODO Update weights
            optimizer.step()

            total_running_loss_train += loss.item()
            total_count += 1

            if (total_count) % config['print_every_n'] == 0:
                print(f'Epoch {epoch}, iteration {total_count}, loss {total_running_loss_train / config["print_every_n"]}')
                total_running_loss_train = 0.0

            if (total_count) % config['validate_every_n'] == 0:
                model.eval()
                total_running_loss_val = 0.0
                total_running_accuracy_val = 0.0

                for val_batch in valloader:
                    ShapeNetPoints.move_batch_to_device(val_batch, device)

                    # TODO Forward pass
                    with torch.no_grad():
                        output = model(val_batch['points'])

                    # TODO Compute loss
                    loss = loss_criterion(output, val_batch['label'])

                    # TODO Compute accuracy
                    accuracy = (output.argmax(dim=1) == val_batch['label']).float().mean()

                    total_running_loss_val += loss.item()
                    total_running_accuracy_val += accuracy.item()

                print(f'Validation loss {total_running_loss_val / len(valloader)}, validation accuracy {total_running_accuracy_val / len(valloader)}')
                
                if accuracy > best_accuracy:
                    torch.save(model.state_dict(), f'exercise_2/runs/{config["experiment_name"]}/model_best.ckpt')
                    best_accuracy = accuracy
                
                model.train()



def main(config):
    """
    Function for training PointNet on ShapeNet
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "exercise_2/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate': learning rate for optimizer
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'validate_every_n': print validation loss and validation accuracy every n iterations
                   'is_overfit': if the training is done on a small subset of data specified in exercise_2/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # Create Dataloaders
    train_dataset = ShapeNetPoints('train' if not config['is_overfit'] else 'overfit')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    val_dataset = ShapeNetPoints('val' if not config['is_overfit'] else 'overfit')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    # Instantiate model
    model = PointNetClassification(ShapeNetPoints.num_classes)

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'exercise_2/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)
