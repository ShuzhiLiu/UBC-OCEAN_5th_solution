import random
from glob import glob
from typing import Any

import albumentations as A
import cv2
import lightning as L
import segmentation_models_pytorch as smp
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from segmentation_models_pytorch.metrics import iou_score, get_stats
from torch.utils.data import DataLoader, Dataset


class SegModelStage1Focus(L.LightningModule):
    def __init__(self, encoder_name, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        self.model = smp.Unet(encoder_name=encoder_name, encoder_weights='imagenet', in_channels=3,
                              classes=num_classes,
                              activation=None)

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.valid_scores = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y.unsqueeze(1) / 255.0)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y.unsqueeze(1) / 255.0)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            eps=1e-6,
        )
        max_epochs = self.trainer.max_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=2e-5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def metric(self, pred, ann, only_foreground=True):
        tp, fp, fn, tn = get_stats(output=pred, target=ann, mode='multiclass', num_classes=self.num_classes,
                                   threshold=None)
        if only_foreground:
            # Ignore background class (assuming it's the first class, index 0)
            tp, fp, fn, tn = tp[1:], fp[1:], fn[1:], tn[1:]
        iou = iou_score(tp, fp, fn, tn, reduction='micro-imagewise')
        return iou


class SegDataset(Dataset):
    def __init__(self, image_dir, train=True):
        image_files = glob(f'{image_dir}/*')
        random.shuffle(image_files)
        if train:
            self.x = image_files[:int(len(image_files) * 0.95)]
            self.x = image_files
            self.transform = A.Compose([
                # A.LongestMaxSize(1536 * 2),
                A.PadIfNeeded(1536, 1536),
                A.CropNonEmptyMaskIfExists(1536, 1536),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5, shift_limit=0.0, scale_limit=0.2, rotate_limit=45, border_mode=0, ),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.2),
                A.ToGray(p=0.2),
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.0),
                        A.ElasticTransform(alpha=3),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=3),
                        A.MedianBlur(blur_limit=3),
                        A.GaussianBlur(blur_limit=3),
                        A.GaussNoise(var_limit=(3.0, 9.0)),
                    ],
                    p=0.15,
                ),
                A.Normalize(),
            ])
        else:
            self.x = image_files[int(len(image_files) * 0.95):]
            self.transform = A.Compose([
                # A.LongestMaxSize(1536 * 2),
                A.PadIfNeeded(1536, 1536),
                A.CropNonEmptyMaskIfExists(1536, 1536),
                A.Normalize(),
            ])

    def __getitem__(self, idx):
        img_path = self.x[idx]
        seg_path = img_path.replace('.jpg', '.png').replace('images', 'anns')
        img = cv2.imread(img_path)
        gt_seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        aug_data = self.transform(image=img, mask=gt_seg)
        img = aug_data['image']
        gt_seg = aug_data['mask']
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        gt_seg = torch.from_numpy(gt_seg).long()
        return img, gt_seg

    def __len__(self):
        return len(self.x)


def train(WANDB_LOG=False,
          encoder_name='tu-seresnextaa101d_32x8d.sw_in12k_ft_in1k_288',
          fast_dev_run=False, EPOCHS=12, batchsize=8, wandb_group_name='train', num_workers=8,
          seg_data_img_path='/kaggle/input/UBC-OCEAN/seg_focus_datasetV1/images',
          work_dir='/kaggle/input/UBC-OCEAN/work_dir/seg',
          n_gpus=1, sync_batchnorm=True):
    if WANDB_LOG:
        wandb_logger = WandbLogger(
            project='UBC',
            entity='sakaku',
            group=f'{wandb_group_name}',
            job_type=f'fold0',
            config=dict(
            )
        )
    else:
        wandb_logger = None

    model = SegModelStage1Focus(encoder_name=encoder_name, num_classes=1)
    train_loader = DataLoader(SegDataset(seg_data_img_path, train=True),
                              batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(SegDataset(seg_data_img_path, train=False),
                             batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    # Define a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=work_dir,
        filename='model-{epoch:02d}',
        save_weights_only=True,
        every_n_epochs=2,  # set the interval, e.g., save every 5 epochs
        # every_n_train_steps=1000,
    )
    trainer = L.Trainer(
        accelerator='gpu',
        devices=n_gpus,
        max_epochs=EPOCHS,
        # max_steps=10000,
        precision='16-mixed',
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_checkpointing=True,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        sync_batchnorm=sync_batchnorm,
        fast_dev_run=fast_dev_run,
        default_root_dir=work_dir,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    wandb.finish()


if __name__ == '__main__':
    torch.set_float32_matmul_precision("medium")  # enable bf16
    torch.multiprocessing.set_sharing_strategy(
        'file_system')  # solve problems of dataloader's multiprocessing
    torch.set_float32_matmul_precision('high')  # for better performance
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    train(WANDB_LOG=False, fast_dev_run=False, EPOCHS=12, batchsize=8, n_gpus=1, wandb_group_name='train',
          seg_data_img_path='/kaggle/input/UBC-OCEAN/seg_focus_datasetV1/images',
          encoder_name='tu-seresnextaa101d_32x8d.sw_in12k_ft_in1k_288',
          work_dir='/kaggle/input/UBC-OCEAN/work_dir/seg_focus',
          num_workers=8)
