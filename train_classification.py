import os
import random
from typing import Any

import cv2
import lightning as L
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchstain
import wandb
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from mmpretrain.models.losses import SeesawLoss
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import f1_score
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class stain_normalizer:
    def __init__(self):
        stain_target = cv2.cvtColor(cv2.imread("/kaggle/input/UBC-OCEAN/target.png"), cv2.COLOR_BGR2RGB)
        self.strain_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
        self.strain_normalizer.fit(stain_target)

        self.trans = transforms.Compose([
            transforms.Resize(size=1536, interpolation=InterpolationMode.BICUBIC),
        ])

    def normalize(self, img: Image.Image):
        to_transform_test = np.array(img)  # RGB
        norm, H, E = self.strain_normalizer.normalize(I=to_transform_test, stains=True)
        norm = norm.astype(np.uint8)
        return Image.fromarray(norm)

    def rand_load(self, filename: str):
        img = Image.open(filename)
        # Check if image size is 3072
        if img.size[0] == 3072:
            img = self.trans(img)
        if random.uniform(0, 1) < 0.15:
            try:
                return self.normalize(img)
            except:
                return img
        else:
            return img


# print('a')
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class UBCDataset(Dataset):
    def __init__(self,
                 transform,
                 fold: int = 0,
                 train_mode: bool = True):
        self.img_prefix = '/kaggle/input/UBC-OCEAN/cropped_wsiv2'
        df1 = pd.read_csv('/kaggle/input/UBC-OCEAN/cropped_wsiV2_split_pseudo_camelyon_tumor_ratio_confidence.csv')
        df1['image_id'] = df1['image_id'].astype(str)
        # remove image id 1289 32035
        df1 = df1[~df1['image_id'].isin(['1289', '32035'])]
        # change image id 15583's label from 3 to 4
        df1.loc[df1['image_id'] == '15583', 'label_int'] = 4
        df1.loc[df1['image_id'] == '15583', 'pseudo_label'] = 4

        df_train = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
        df_train['image_id'] = df_train['image_id'].astype(str)
        df_ori = df1[df1['image_id'].isin(df_train['image_id'])]
        df_ex = df1[~df1['image_id'].isin(df_train['image_id'])]

        def set_pseudo_label(row):
            confidence = row['label_confidence']
            if confidence >= 0.6:
                return row['label_int']
            elif confidence >= 0.3:
                return 255
            else:
                return 5

        # keep top 10 and label_confidence >= 0.6
        dfs = []
        for image_id, group_df in df_ori.groupby('image_id'):
            group_df = group_df.sort_values(by='label_confidence', ascending=False)
            # dfs.append(group_df)
            if len(group_df) > 10:
                group_df_top10 = group_df[:10].copy()
                group_df_top10['pseudo_label'] = group_df_top10['label_int']

                group_df_remain = group_df[10:].copy()
                group_df_remain['pseudo_label'] = group_df_remain.apply(set_pseudo_label, axis=1)
                group_df_new = pd.concat([group_df_top10, group_df_remain])
                assert len(group_df_new) == len(group_df)
                dfs.append(group_df_new)
            else:
                group_df['pseudo_label'] = group_df['label_int']
                dfs.append(group_df)
        df_new = pd.concat(dfs)
        df1 = pd.concat([df_new, df_ex])

        df1 = df1[df1['pseudo_label'] != 255]
        df1_other = df1[df1['pseudo_label'] == 5]
        # df1_other = df1[df1.apply(lambda x: x['pseudo_label'] == 5 and 'tumer' not in x['image_id'], axis=1)]
        df1 = df1[df1['pseudo_label'] != 5]
        # df1 = df1[df1.apply(lambda x: x['pseudo_label'] != 5 or 'tumer' in x['image_id'], axis=1)]
        df_hub_other = pd.read_csv('/kaggle/input/hubmap-organ-segmentation/train_split.csv')

        if train_mode:
            pass
            # df1 = df1[df1['split'] != fold]
            # df_hub_other = df_hub_other[df_hub_other['split'] != fold]
        else:
            df1 = df1[df1['split'] == fold]
            df_hub_other = df_hub_other[df_hub_other['split'] == fold]

        self.df1_other_list = df1_other.to_dict("records")
        self.data_list = df1.to_dict("records")
        self.transform = transform
        self.image_ids = df1['image_id'].unique()
        self.image_ids_tma = df1[df1['is_tma']]['image_id'].unique()
        # image_id to image_file dict
        self.image_id_to_file = df1.groupby('image_id')['image_file'].apply(list).to_dict()
        self.image_id_to_label = df1.groupby('image_id')['label_int'].apply(set).apply(list).to_dict()
        self.hub_other_ids = df_hub_other['id'].unique().tolist()
        self.train_mode = train_mode
        self.tma_trans1 = transforms.Compose([
            transforms.RandomCrop(size=(3072, 3072), pad_if_needed=True),
            # transforms.Resize(size=(1536, 1536), interpolation=InterpolationMode.BICUBIC),
        ])

        self.stain_norm = stain_normalizer()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, id):
        image_id = self.image_ids[id]
        if random.uniform(0, 1) < 0.1:
            # use hubmap other images
            if random.uniform(0, 1) < 0.5:
                hub_other_id = random.choice(self.hub_other_ids)
                filename = f"/kaggle/input/hubmap-organ-segmentation/train_images/{hub_other_id}.tiff"
                img = self.stain_norm.rand_load(filename)
                img = self.tma_trans1(img)
            else:
                df1_other_sample = random.choice(self.df1_other_list)
                filename = df1_other_sample['image_file']
                img = self.stain_norm.rand_load(os.path.join(self.img_prefix, filename))
            label = 5
        else:
            if image_id in self.image_ids_tma:
                filename = f"/kaggle/input/UBC-OCEAN_NEW/train_images/{image_id}.png"
                label = self.image_id_to_label[image_id][0]
                img = self.stain_norm.rand_load(filename)
                img = self.tma_trans1(img)
            else:
                filename = random.choice(self.image_id_to_file[image_id])
                label = self.image_id_to_label[image_id][0]
                img = self.stain_norm.rand_load(os.path.join(self.img_prefix, filename))
        img = self.transform(img)
        return img, label


# parameters
model_candidates = ['coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',
                    'convnextv2_large.fcmae_ft_in22k_in1k_384',
                    'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384',
                    'convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384',
                    'tf_efficientnet_l2.ns_jft_in1k', ]


class UBCClsModel(L.LightningModule):
    def __init__(self, model_name='convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384', lr=1e-4, lr_min=2e-5,
                 lr_decay=0.8):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True,
                                       num_classes=6,
                                       drop_rate=0.1,
                                       drop_path_rate=0.1,
                                       )
        self.model.set_grad_checkpointing()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # cc, ec, hgsc, lgsc, mc, other
        self.seesaw_fn = SeesawLoss(num_classes=6)
        self.valid_labels = []
        self.valid_logits = []
        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.alpha = 0.2

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        loss += self.seesaw_fn(logits, y)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        grouped_optimizer_params = self.get_optimizer_grouped_parameters(
            self.model,
            self.lr, 0.01,
            self.lr_decay
        )
        optimizer = AdamW(
            grouped_optimizer_params,
            lr=self.lr,
            eps=1e-6,
        )
        max_epochs = self.trainer.max_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=self.lr_min)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self.model(x)
        self.valid_logits.append(logits)
        self.valid_labels.append(y)
        return None

    def on_validation_epoch_end(self) -> None:
        self.valid_logits = torch.cat(self.valid_logits, dim=0)
        self.valid_labels = torch.cat(self.valid_labels, dim=0)
        loss = self.loss_fn(self.valid_logits, self.valid_labels)
        loss += self.seesaw_fn(self.valid_logits, self.valid_labels)
        self.log('val/loss', loss, prog_bar=True)
        # f1 score for each class
        self.valid_pred_labels = torch.argmax(self.valid_logits, dim=1)
        f1_cc = f1_score(torch.where(self.valid_labels == 0, 1, 0).cpu().numpy().flatten(),
                         torch.where(self.valid_pred_labels == 0, 1, 0).cpu().numpy().flatten())
        f1_ec = f1_score(torch.where(self.valid_labels == 1, 1, 0).cpu().numpy().flatten(),
                         torch.where(self.valid_pred_labels == 1, 1, 0).cpu().numpy().flatten())
        f1_hgsc = f1_score(torch.where(self.valid_labels == 2, 1, 0).cpu().numpy().flatten(),
                           torch.where(self.valid_pred_labels == 2, 1, 0).cpu().numpy().flatten())
        f1_lgsc = f1_score(torch.where(self.valid_labels == 3, 1, 0).cpu().numpy().flatten(),
                           torch.where(self.valid_pred_labels == 3, 1, 0).cpu().numpy().flatten())
        f1_mc = f1_score(torch.where(self.valid_labels == 4, 1, 0).cpu().numpy().flatten(),
                         torch.where(self.valid_pred_labels == 4, 1, 0).cpu().numpy().flatten())
        self.log('val/f1_cc', f1_cc, prog_bar=True, sync_dist=True)
        self.log('val/f1_ec', f1_ec, prog_bar=True, sync_dist=True)
        self.log('val/f1_hgsc', f1_hgsc, prog_bar=True, sync_dist=True)
        self.log('val/f1_lgsc', f1_lgsc, prog_bar=True, sync_dist=True)
        self.log('val/f1_mc', f1_mc, prog_bar=True, sync_dist=True)

        self.valid_logits = []
        self.valid_labels = []

    @staticmethod
    def get_optimizer_grouped_parameters(
            model,
            learning_rate, weight_decay,
            layerwise_learning_rate_decay
    ):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "head" in n],
                "weight_decay": 0.0,
                "lr": learning_rate,
            },
        ]
        layers = list(model.stem) + list(model.stages)
        layers.reverse()
        lr = learning_rate
        for layer in layers:
            lr *= layerwise_learning_rate_decay
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
        optimizer_grouped_parameters.reverse()
        return optimizer_grouped_parameters


def train(FOLD=0,
          EPOCHS=36,
          BATCH=8,
          n_gpus=2,
          num_workers=8,
          WANDB_LOG=False,
          fast_dev_run=False,
          wandb_group_name='baseline',
          model_name='convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384', lr=1e-4, lr_min=2e-5,
          lr_decay=0.8, accumulate_grad_batches=4,
          resize_size=1024, ):
    # ==== dataset, dataloader
    trans_train = transforms.Compose([
        transforms.Resize(size=resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        transforms.RandomErasing(),
    ])
    trans_test = transforms.Compose([
        transforms.Resize(size=resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    train_set = UBCDataset(transform=trans_train, fold=FOLD, train_mode=True)
    test_set = UBCDataset(transform=trans_test, fold=FOLD, train_mode=False)
    train_loader = DataLoader(
        dataset=train_set, num_workers=num_workers, pin_memory=False, batch_size=BATCH, shuffle=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        dataset=test_set, num_workers=4, pin_memory=True, batch_size=8, shuffle=False,
    )
    # ====== model
    model = UBCClsModel(model_name=model_name, lr=lr, lr_min=lr_min, lr_decay=lr_decay)

    if WANDB_LOG:
        wandb_logger = WandbLogger(
            project='UBC',
            entity='sakaku',
            group=f'{wandb_group_name}',
            job_type=f'fold_{FOLD}',
            config=dict(
                model_name=model_name, lr=lr, lr_min=lr_min, lr_decay=lr_decay,
                epochs=EPOCHS, batch_size=BATCH, accumulate_grad_batches=accumulate_grad_batches,
            )
        )
    else:
        wandb_logger = None

    # Define a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'/kaggle/input/UBC-OCEAN/work_dir/baselinecls/{wandb_group_name}',
        filename='model-{epoch:02d}' + f'fold_{FOLD}',
        save_weights_only=True,
        every_n_epochs=2
    )
    trainer = L.Trainer(
        accelerator='gpu',
        devices=n_gpus,
        max_epochs=EPOCHS,
        precision='bf16-mixed',
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_checkpointing=True,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        sync_batchnorm=False,
        fast_dev_run=fast_dev_run,
        default_root_dir=f'/kaggle/input/UBC-OCEAN/work_dir/baselinecls/{wandb_group_name}',
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    wandb.finish()


if __name__ == '__main__':
    torch.set_float32_matmul_precision("medium")  # enable bf16
    torch.multiprocessing.set_sharing_strategy(
        'file_system')  # solve problems of dataloader's multiprocessing?
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    FOLDS = [0, ]
    for fold in FOLDS:
        seed_everything(42)
        train(FOLD=fold,
              n_gpus=1,
              EPOCHS=128,
              BATCH=32,
              num_workers=8,
              WANDB_LOG=True,
              wandb_group_name='Ex_Pseudo_Base_StainNorm_RM_FullData',
              model_name='convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384', lr=1e-4, lr_min=1e-5,
              lr_decay=0.7, accumulate_grad_batches=8,
              fast_dev_run=False,
              resize_size=1536)