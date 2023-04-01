import pathlib
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from common.args import Args
from data import transforms as T
from model import Model
import scipy.io as sio
from common.subsample import create_mask_for_mask_type
import math
from unet_model import UnetModel
from pytorch_lightning.callbacks import LearningRateMonitor, DeviceStatsMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from common import evaluate
import timeit

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, target, fname, slice):
        image = T.to_tensor(target.astype(complex))
        k_space = T.fft2c_new(image)

        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask = T.apply_mask(k_space, self.mask_func, seed)
        "Add noise"
        # tau = 0.0042
        # masked_kspace = masked_kspace + (torch.randn(masked_kspace.shape))/math.sqrt(2)* tau
        zf = T.ifft2c_new(masked_kspace)
        dirty = T.complex_abs(zf).type(torch.FloatTensor)

        target = T.to_tensor(target.astype(np.float32))

        return dirty, target, fname, slice,masked_kspace,mask


class Unet_Model(Model):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.unetwdsr = UnetModel(in_chans=1,
                                  out_chans=1,
                                  chans=hparams.num_chans,
                                  num_pool_layers=hparams.num_pools,
                                  drop_prob=hparams.drop_prob,
                                  )

    def forward(self, input):
        start = timeit.default_timer()
        output = self.unetwdsr(input)
        output = torch.add(input, output)
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return output

    def training_step(self, batch, batch_idx):
        dirty, target, _, _,_,_ = batch
        dirty, mean, std = T.normalize_instance(dirty, eps=1e-110)
        target = T.normalize(target, mean, std, eps=1e-110)
        output = self.forward(dirty)
        loss = F.l1_loss(output, target)
        logs = {'loss': loss.item()}
        return dict(loss=loss, log=logs)

    def validation_step(self, batch, batch_idx):
        dirty, target, fname, slice,_,_ = batch
        dirty, mean, std = T.normalize_instance(dirty, eps=1e-110)
        target = T.normalize(target, mean, std, eps=1e-110)
        output = self.forward(dirty)
        logSNR = evaluate.snr(evaluate.to_log((target.squeeze(1)*std + mean).cpu().numpy()),
                              evaluate.to_log((output.squeeze(1)*std + mean).cpu().numpy()))
        self.log("val_logSNR", logSNR, on_epoch=True, batch_size=1)
        return {
            'fname': fname,
            'slice': slice,
            'ZF': (dirty.squeeze(1)).cpu().numpy(),
            'output': (output.squeeze(1) *std + mean).cpu().numpy(),
            'target': (target.squeeze(1) *std + mean).cpu().numpy(),
            'val_loss': F.l1_loss(output, target),
        }

    def test_step(self, batch, batch_idx):
        dirty, target, fname, slice,masked_kspace,mask= batch
        dirty, mean, std = T.normalize_instance(dirty, eps=1e-110)
        output = self.forward(dirty)
        return {
            'fname': fname,
            'slice': slice,
            'test_ZF': (dirty.squeeze(1)*std+mean).cpu().numpy(),
            'test_output': output.squeeze(1)*std+mean.cpu().numpy(),
            'test_target': target.squeeze(1).cpu().numpy(),
            'test_loss': F.l1_loss(output, target),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.hparams.lr_step_size, self.hparams.lr_gamma)
        return [optim], [scheduler]

    def train_data_transform(self):
        mask = create_mask_for_mask_type(self.hparams.mask_type, self.hparams.center_fractions,
                                         self.hparams.accelerations)
        return DataTransform(self.hparams.resolution, self.hparams.challenge, mask, use_seed=True)

    def val_data_transform(self):
        mask = create_mask_for_mask_type(self.hparams.mask_type, self.hparams.center_fractions,
                                         self.hparams.accelerations)
        return DataTransform(self.hparams.resolution, self.hparams.challenge, mask, use_seed=True)

    def test_data_transform(self):
        mask = create_mask_for_mask_type(self.hparams.mask_type, self.hparams.center_fractions,
                                         self.hparams.accelerations)
        return DataTransform(self.hparams.resolution, self.hparams.challenge, mask, use_seed=True)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--num_pools', type=int, default=4, help='Number of U-Net pooling layers')
        parser.add_argument('--drop_prob', type=float, default=0.0, help='Dropout probability')
        parser.add_argument('--num_chans', type=int, default=32, help='Number of U-Net channels')
        parser.add_argument('--batch_size', default=1, type=int, help='Mini batch size')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--lr-step-size', type=int, default=1000, help='Period of learning rate decay')
        parser.add_argument('--lr-gamma', type=float, default=0.1, help='Multiplicative factor of learning rate decay')
        parser.add_argument('--gradient_clip_val', type=float, default=1e-2)
        parser.add_argument('--weight-decay', type=float, default=0.,help='Strength of weight decay regularization')
        return parser

lr_monitor = LearningRateMonitor(logging_interval='epoch')
device_stats = DeviceStatsMonitor()
def create_trainer(args, logger):
    # checkpoint_callback = ModelCheckpoint(monitor="val_loss",str(args.checkpoint), filename=args.exp+'_{epoch:02d}',
    # save_top_k=-1,mode="min")
    # early_stop_callback = EarlyStopping(monitor="val_logSNR", min_delta=0.0001, patience=20, verbose=False, mode="max",
    #                                     strict=True, check_finite=True, check_on_train_epoch_end=False)
    return Trainer(
        logger=logger,
        max_epochs=args.num_epochs,
        gpus=args.gpus,
        strategy='ddp',
        # auto_lr_find=args.lr,
        # auto_scale_batch_size="binsearch",
        # callbacks=[checkpoint_callback],
        # callbacks=[early_stop_callback],
        #gradient_clip_val=args.gradient_clip_val,
        #precision=16,
        # check_val_every_n_epoch=1,
        val_check_interval=1.,
        #callbacks=[lr_monitor, device_stats],
        #log_every_n_steps=10
    )



def main(args):
    if args.mode == 'train':
        load_version = 0 if args.resume else None

        logger = TensorBoardLogger(save_dir=args.exp_dir, name=args.exp, version=load_version)
        trainer = create_trainer(args, logger)
        model = Unet_Model(args)

        pytorch_total_params = sum (p.numel() for p in model.parameters())
        print('Total number of params:', pytorch_total_params)

        # trainer.tune(model)
        trainer.fit(model, ckpt_path=args.checkpoint)
        # checkpoint_callback = ModelCheckpoint(monitor="val_loss",str(args.checkpoint), filename=args.exp + '_{epoch:02d}_')
        # checkpoint_callback.best_model_path
    else:
        args.mode == 'test'
        assert args.checkpoint is not None
        model = Unet_Model.load_from_checkpoint(str(args.checkpoint), data_path=(args.data_path))
        model.hparams.sample_rate = 1.
        model.hparams.rec_ext = args.rec_ext
        model.hparams.rec2_ext = args.rec2_ext
        model.hparams.checkpoint = args.checkpoint
        model.hparams.exp = args.exp
        model.hparams.scname_test = args.scname_test
        trainer = create_trainer(args, logger=True)
        trainer.test(model)


if __name__ == '__main__':
    parser = Args()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--exp-dir', type=pathlib.Path, default='experiments',
                        help='Path where model and results should be saved')
    parser.add_argument('--exp', type=str, help='Name of the experiment')
    parser.add_argument('--checkpoint', type=pathlib.Path,
                        help='Path to pre-trained model. Use with --mode test')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. ')
    parser.add_argument('--num_blocks', help='Number of residual blocks in networks.', default=16, type=int)
    parser.add_argument('--num_residual_units', help='Number of residual units in networks.', default=32, type=int)
    parser.add_argument('--width_multiplier', help='Width multiplier inside residual blocks.', default=4, type=float)
    parser.add_argument('--temporal_size', help='Number of frames for burst input.', default=None, type=int)
    parser.add_argument('--scale', type=int, default=1)
    parser.set_defaults(image_mean=0.5)
    parser = Unet_Model.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
