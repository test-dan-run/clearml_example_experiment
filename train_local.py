from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from argparse import ArgumentParser
from typing import Tuple
from model import LightningMNISTClassifier

def prepare_data(dataset_path: str) -> Tuple[MNIST]:

    # transforms for images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # load datasets and prepare transforms for a subset
    mnist_train = MNIST(root=dataset_path, train=True, download=False, transform=transform)
    mnist_train = [mnist_train[i] for i in range(2200)]
    mnist_train, mnist_val = random_split(mnist_train, [2000, 200])

    mnist_test = MNIST(root=dataset_path, train=False, download=False, transform=transform)
    mnist_test = [mnist_test[i] for i in range(3000, 3500)]

    return mnist_train, mnist_val, mnist_test

def train(
    model: LightningMNISTClassifier,
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader, 
    test_dataloader: DataLoader, 
    num_epochs: int = 30,
    num_gpus: int = 1,
    ) -> None:

    # Setup tensorboard logger
    tb_logger = TensorBoardLogger(save_dir='tb_logs')

    # Setup training callbacks
    lr_logger = LearningRateMonitor()
    early_stop = EarlyStopping('val_loss', mode='min', patience=5)
    checkpoint_callback = ModelCheckpoint(
        filename='mnist_{epoch}-{val_loss:.2f}',
        monitor='val_loss', mode='min', save_top_k=1
        )
    
    # PyTorch Lightning trainer, which will help manage the training process
    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus, 
        logger=tb_logger,
        callbacks=[lr_logger, early_stop, checkpoint_callback],
        strategy=DDPPlugin(find_unused_parameters=False) if num_gpus > 1 else None,
        )

    # train
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # test
    trainer.test(dataloaders=test_dataloader)

def main(
    dataset_path: str, 
    batch_size: int = 64,
    num_gpus: int = 1,
    num_epochs: int = 30,
    lr: float = 1e-3,
    dropout_prob: float = 0.1,
    ):

    # load data and initialise as dataloaders
    train_dataset, val_dataset, test_dataset = prepare_data(dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # initialise model
    model = LightningMNISTClassifier(lr=lr, dropout_prob=dropout_prob)

    # run training pipeline
    train(model, train_loader, val_loader, test_loader, num_epochs=num_epochs, num_gpus=num_gpus)

if __name__ == '__main__':
    parser = ArgumentParser(description='Sample experiment with mnist')
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-g', '--gpus', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-l', '--lr', type=float, default=1e-3)
    parser.add_argument('-d', '--dropout_prob', type=float, default=0.1)
    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_gpus=args.gpus,
        num_epochs=args.epochs,
        lr=args.lr,
        dropout_prob=args.dropout_prob
    )
