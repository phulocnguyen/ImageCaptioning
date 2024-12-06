from typing import Any, Dict, Optional, Tuple

import torch
import rootutils
import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dataset import init_dataset
from src.data.preprocessing import PreprocessingDataset


class ImageCaptionDataModule(L.LightningDataModule):
    """
    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split,  process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "./data",
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 2,
        pin_memory: bool = False,
        dataset_name: str = 'flickr8k',
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return self.hparams.n_classes

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = init_dataset(self.hparams.dataset_name,
                                   data_dir=self.hparams.data_dir)

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            print('Number of sequences in Train-Val-Test Dataset:',
                  len(self.data_train), len(self.data_val),
                  len(self.data_test))

            print('=' * 10, 'preprocessing train dataset', '=' * 10)
            self.data_train = PreprocessingDataset(
                dataset=self.data_train, dataset_dir=dataset.dataset_dir)

            print('=' * 10, 'validation train dataset', '=' * 10)
            self.data_val = PreprocessingDataset(
                dataset=self.data_val, dataset_dir=dataset.dataset_dir)

            print('=' * 10, 'test train dataset', '=' * 10)
            self.data_test = PreprocessingDataset(
                dataset=self.data_test, dataset_dir=dataset.dataset_dir)

            print('Number of sequences in Train-Val-Test PreprocessedDataset:',
                  len(self.data_train), len(self.data_val),
                  len(self.data_test))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "data")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="flickr8k.yaml")
    def main(cfg: DictConfig):
        print(cfg)

        datamodule: ImageCaptionDataModule = hydra.utils.instantiate(
            cfg, data_dir=f"{root}/data")
        datamodule.setup()

        train_dataloader = datamodule.train_dataloader()
        print('train_dataloader:', len(train_dataloader))

        batch = next(iter(train_dataloader))
        image, captions = batch
        print(image.shape, captions.shape)

        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid
        image = make_grid(image[:25], nrow=5)

        plt.imshow(image.moveaxis(0, 2))
        plt.show()

    main()