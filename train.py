from argparse import ArgumentParser
from pathlib import Path
import sys

import torch
from torch import nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.models import get_model
from src.data import get_data_loaders


# define globally-scoped variables
train_batch_idx = -1
max_val_acc = 0


def run_training(
    model: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    ckpts_path: Path,
    device: str,
):
    # Training loop
    for epoch_idx in tqdm(range(num_epochs), leave=True):
        # Training epoch
        model.train()
        training_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch_idx=epoch_idx,
            device=device,
            dl_train=dl_train,
        )

        # Validation epoch
        model.eval()
        validation_epoch(
            model=model,
            epoch_idx=epoch_idx,
            device=device,
            dl_val=dl_val,
            ckpt_path=Path(ckpts_path),
        )


def training_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    epoch_idx: int,
    device: torch.device,
    dl_train: DataLoader,
):
    global train_batch_idx
    for (train_batch_idx, batch) in enumerate(
        tqdm(dl_train, leave=False), start=train_batch_idx + 1
    ):
        imgs, targets = batch
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Compute loss
        preds = model(imgs)
        loss = loss_fn(preds, targets)

        # Take optimization step
        loss.backward()
        optimizer.step()
        model.zero_grad()

        # Log loss
        wandb.log({
            f'Train/{loss_fn.__class__.__name__}': loss,
            "epoch": epoch_idx,
            'batch_idx': train_batch_idx
        })

    if torch.isnan(loss):
        sys.exit('Loss is NaN. Exiting...')


@torch.no_grad()
def validation_epoch(
    model: nn.Module,
    epoch_idx: int,
    device: torch.device,
    dl_val: DataLoader,
    ckpt_path: Path,
):
    accuracies = []
    losses = []

    for batch_idx, batch in enumerate(tqdm(dl_val, leave=False)):
        imgs, targets = batch
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Compute loss
        preds = model(imgs)
        loss = loss_fn(preds, targets)
        losses.append(loss)

        # Compute accuracy
        preds = preds.argmax(dim=1)
        acc = (preds == targets).float().mean()
        accuracies.append(acc)

    mean_acc = torch.stack(accuracies).mean().cpu().numpy()
    mean_loss = torch.stack(losses).mean().cpu().numpy()

    # Log validation metrics
    wandb.log({
        'Val/Accuracy': mean_acc,
        "epoch": epoch_idx,
    })
    wandb.log({
        f'Val/{loss_fn.__class__.__name__}': mean_loss,
        "epoch": epoch_idx,
    })

    ckpt_path.mkdir(parents=True, exist_ok=True)
    suffix = '.pth'

    # Always save last model
    torch.save(model.state_dict(),
               ckpt_path / f'{wandb.run.id}_last{suffix}')

    # Save copy if accuracy increased
    global max_val_acc
    if mean_acc > max_val_acc:
        max_val_acc = mean_acc

        prefix = f'{wandb.run.id}_ep'

        # Remove previously created checkpoint(s)
        for p in ckpt_path.glob(f'{prefix}*{suffix}'):
            p.unlink()

        # Save checkpoint
        torch.save(model.state_dict(),
                   ckpt_path / f'{prefix}{epoch_idx}{suffix}')


if __name__ == '__main__':
    parser = ArgumentParser()

    # Model
    parser.add_argument(
        '--model_name',
        help='The name of the model to use for the classifier.',
        default='resnet18',
    )
    parser.add_argument(
        '--model_weights',
        help='The pretrained weights to load. If None, the weights are '
        'randomly initialized. See also '
        'https://pytorch.org/vision/stable/models.html.',
        default=None
    )

    # Checkpoints
    parser.add_argument(
        '--ckpts_path',
        default='./ckpts',
        help='The directory to save checkpoints.'
    )
    parser.add_argument(
        '--load_ckpt',
        default=None,
        help='The path to load model checkpoint weights from.'
    )

    # Data path
    parser.add_argument(
        '--data_path',
        default='data/PokemonGen1',
        help='Path to the dataset',
    )

    # K-Fold args
    parser.add_argument(
        '--num_folds',
        default=5,
        help='The number of folds to use for cross-validation.',
        type=int
    )
    parser.add_argument(
        '--val_fold',
        default=0,
        help='The index of the validation fold. '
        'If None, all folds are used for training.',
        type=int
    )

    # Data loader args
    parser.add_argument(
        '--batch_size',
        default=32,
        help='The training batch size.',
        type=int
    )
    parser.add_argument(
        '--val_batch_size',
        default=32,
        help='The validation batch size.',
        type=int
    )
    parser.add_argument(
        '--num_workers',
        default=8,
        help='The number of workers to use for data loading.',
        type=int
    )

    # Data transform args
    parser.add_argument(
        '--size',
        default=224,
        help='The size to use in the data transform pipeline.',
        type=int,
    )

    # Optimizer args
    parser.add_argument(
        '--lr',
        default=0.01,
        help='The learning rate.',
        type=float
    )
    parser.add_argument(
        '--momentum',
        default=0,
        help='The momentum.',
        type=float
    )
    parser.add_argument(
        '--weight_decay',
        default=0,
        help='The weight decay.',
        type=float
    )

    # Train args
    parser.add_argument(
        '--num_epochs',
        default=20,
        help='The number of epochs to train.',
        type=int
    )

    # Log args
    parser.add_argument(
        '--wandb_entity',
        default='YOUR_WANDB_USER_NAME',
        help='Weights and Biases entity.',
    )
    parser.add_argument(
        '--wandb_project',
        help='Weights and Biases project.'
    )

    args = parser.parse_args()

    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=vars(args)
    )

    model = get_model(
        name=args.model_name,
        weights=args.model_weights,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    model = model.to(device)

    # Load checkpoint
    if args.load_ckpt is not None:
        model.load_state_dict(torch.load(args.load_ckpt))

    # Define optimizer
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Get data loaders
    dl_train, dl_val = get_data_loaders(
        data_path=args.data_path,
        size=args.size,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        num_folds=args.num_folds,
        val_fold=args.val_fold,
    )

    # Run training
    run_training(
        model=model,
        dl_train=dl_train,
        dl_val=dl_val,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        ckpts_path=args.ckpts_path,
        device=device,
    )
