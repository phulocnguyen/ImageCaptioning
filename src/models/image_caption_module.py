from typing import Any, Dict, Optional, Tuple
from lightning.pytorch.utilities.types import STEP_OUTPUT

import wandb
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.text import BLEUScore
from pickle import load
import os.path as osp

from src.models.components import ImageCaptionNet
from src.utils.decode import greedy_search, batch_greedy_search, beam_search_decoding


class ImageCaptionModule(LightningModule):

    def __init__(
        self,
        net: ImageCaptionNet,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        dataset_dir: str = 'data/flickr8k',
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        vocab_size_path = osp.join(dataset_dir, 'vocab_size.pkl')
        if not osp.exists(vocab_size_path):
            raise ValueError(
                "weight_embedding_path is not exist. Please check path or run datamodule to prepare"
            )

        with open(vocab_size_path, "rb") as file:
            vocab_size = load(file)

        id2word_path = osp.join(dataset_dir, 'id2word.pkl')
        with open(id2word_path, "rb") as file:
            self.id2word = load(file)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=vocab_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=vocab_size)
        self.test_acc = Accuracy(task="multiclass", num_classes=vocab_size)
        self.test_bleu1 = BLEUScore(n_gram=1)
        self.test_bleu2 = BLEUScore(n_gram=2)
        self.test_bleu3 = BLEUScore(n_gram=3)
        self.test_bleu4 = BLEUScore(n_gram=4)
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.batch = {
            'train': [],
            'valid': [],
            'test': [],
        }

    def forward(self, image: torch.Tensor,
                sequence: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(image, sequence)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        image, captions = batch

        images = []
        sequences = []
        targets = []

        # captions: batch, 5, max_length
        for n_cap in range(captions.shape[1]):
            for i in range(1, captions.shape[2]):
                id = captions[:, n_cap, i] != 0  # not learn output = <pad>
                if not id.any(): break  # <pad> all

                source, target = captions[id, n_cap, :i], captions[id, n_cap,
                                                                   i]
                source = torch.nn.functional.pad(source,
                                                 (captions.shape[2] - i, 0),
                                                 value=0)
                images.append(image[id])
                sequences.append(source)
                targets.append(target)

        images = torch.cat(images, dim=0)
        sequences = torch.cat(sequences, dim=0)
        targets = torch.cat(targets, dim=0)

        logits = self.forward(images, sequences)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, targets

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss",
                 self.train_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("train/acc",
                 self.train_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        return
        self.inference(mode='train')

    def calculate_bleu(self, batch, metrics, search: str = 'greedy'):
        if search == 'greedy':
            search = greedy_search
        elif search == 'batch_greedy':
            search = batch_greedy_search
        elif search == 'beam':
            search = beam_search_decoding
        # elif search == 'batch_beam':
        #     search = batch_beam_search_decoding
        else:
            raise NotImplementedError(f"unknown search: {search}")
        # calculate bleu score
        with torch.no_grad():
            images, captions = batch
            batch_length = images.shape[0]
            for i in range(batch_length):
                e_image = images[i]
                e_captions = captions[i]
                
                targets = []
                for caption in e_captions:
                    caption = [
                        self.id2word[id.cpu().item()] for id in caption if id != 0
                    ]
                    caption = ' '.join(caption[1:-1])
                    targets.append(caption)
                pred = search(model=self.net, images=e_image.unsqueeze(0))
                for metric in metrics:
                    metric.update(pred, [targets])
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss",
                 self.val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("val/acc",
                 self.val_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best",
                 self.val_acc_best.compute(),
                 sync_dist=True,
                 prog_bar=True)

        self.inference(mode='val')

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                  batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.calculate_bleu(batch, [self.test_bleu1, self.test_bleu2, self.test_bleu3, self.test_bleu4])
        
        self.log("test/loss",
                 self.test_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("test/acc",
                 self.test_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("test/bleu1",
                 self.test_bleu1,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("test/bleu2",
                 self.test_bleu2,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("test/bleu3",
                 self.test_bleu3,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("test/bleu4",
                 self.test_bleu4,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.inference(mode='test')

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass
        # if self.hparams.compile and stage == "fit":
        #     self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any,
                           batch_idx: int) -> None:
        if batch_idx == 0:
            self.store_data(batch, mode='train')

    def on_validation_batch_end(self,
                                outputs: STEP_OUTPUT | None,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        if batch_idx == 0:
            self.store_data(batch, mode='val')

    def on_test_batch_end(self,
                          outputs: STEP_OUTPUT | None,
                          batch: Any,
                          batch_idx: int,
                          dataloader_idx: int = 0) -> None:
        if batch_idx == 0:
            self.store_data(batch, mode='test')

    def store_data(self, batch: Any, mode: str):
        self.batch[mode] = batch

    def inference(self, mode: str, search: str = 'greedy'):
        if search == 'greedy':
            search = greedy_search
        elif search == 'batch_greedy':
            search = batch_greedy_search
        elif search == 'beam':
            search = beam_search_decoding
        # elif search == 'batch_beam':
        #     search = batch_beam_search_decoding
        else:
            raise NotImplementedError(f"unknown search: {search}")

        preds = search(model=self.net, images=self.batch[mode][0])

        data = []
        for pred, img, captions in zip(preds, self.batch[mode][0],
                                       self.batch[mode][1]):
            targets = []
            for caption in captions:
                caption = [
                    self.id2word[id.cpu().item()] for id in caption if id != 0
                ]
                caption = ' '.join(caption[1:-1])
                targets.append(caption)

            targets = ' | '.join(targets)
            data.append([wandb.Image(img), pred, targets])

        self.logger.log_table(key=f'{mode}/infer',
                              columns=['image', 'pred', 'caption'],
                              data=data)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    import rootutils
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name=".yaml")
    def main(cfg: DictConfig):
        print(cfg)

        module: ImageCaptionModule = hydra.utils.instantiate(cfg)

        sequences = torch.randint(0, 100, (20, 2))
        images = torch.randn(2, 3, 299, 299)
        out = module.net(images, sequences)
        print(out.shape)

    main()