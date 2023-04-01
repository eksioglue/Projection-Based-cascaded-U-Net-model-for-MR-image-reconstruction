
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DistributedSampler, DataLoader
from common.utils import save_reconstructions, save_ZF, save_target
from common import evaluate
from data.mri_data import SliceData
# from torch.utils.tensorboard import SummaryWriter




class Model(pl.LightningModule):
    """
    Abstract super class for Deep Learning based reconstruction models.
    This is a subclass of the LightningModule class from pytorch_lightning, with
    some additional functionality specific to fastMRI:
        - fastMRI data loaders
        - Evaluating reconstructions
        - Visualization
        - Saving test reconstructions

    To implement a new reconstruction model, inherit from this class and implement the
    following methods:
        - train_data_transform, val_data_transform, test_data_transform:
            Create and return data transformer objects for each data split
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation and testing respectively
        - configure_optimizers:
            Create and return the optimizers
    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def _create_data_loader(self, data_transform, data_partition, sample_rate=None):
        sample_rate = sample_rate or self.hparams.sample_rate

        dataset = SliceData(
            root=self.hparams.data_path / f'{data_partition}',
            transform=data_transform,
            sample_rate=1,
            #challenge=self.hparams.challenge
        )
        #sampler = DistributedSampler(dataset)
        return DataLoader(
            dataset=dataset,
            #shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=8,
            pin_memory=True,
            sampler=None,
        )

    def train_data_transform(self):
        raise NotImplementedError


    def train_dataloader(self):
        return self._create_data_loader(self.train_data_transform(), data_partition=self.hparams.scname_train)

    def val_data_transform(self):
        raise NotImplementedError


    def val_dataloader(self):
        return self._create_data_loader(self.val_data_transform(), data_partition=self.hparams.scname_val)

    def test_data_transform(self):
        raise NotImplementedError


    def test_dataloader(self):
        return self._create_data_loader(self.test_data_transform(), data_partition=self.hparams.scname_test, sample_rate=1.)



    def _evaluate(self, val_logs):

        # self.writer = SummaryWriter('summary/'+self.hparams.Delta_t+"/"+self.hparams.exp)

        losses = []
        Zero_fills = defaultdict(list)
        outputs = defaultdict(list)
        targets = defaultdict(list)
        for log in val_logs:
            losses.append(log['val_loss'].cpu().numpy())
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
                Zero_fills[fname].append((slice, log['ZF'][i]))



        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[], SNR=[], log_SNR=[], std_nmse=[],
                            std_ssim=[], std_psnr=[], std_logSNR=[], std_SNR=[],
                            zf_nmse=[], zf_ssim=[], zf_psnr=[],zf_logSNR=[], zf_SNR=[])

        for fname in outputs:
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            Zero_filling = np.stack([zf for _, zf in sorted(Zero_fills[fname])])

            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['std_nmse'].append(np.std(metrics['nmse']))
            metrics['ssim'].append(evaluate.ssim(target, output))
            metrics['std_ssim'].append(np.std(metrics['ssim']))

            metrics['psnr'].append(evaluate.psnr(target, output))
            metrics['std_psnr'].append((np.std(metrics['psnr'])))

            metrics['log_SNR'].append(evaluate.snr(evaluate.to_log(target), evaluate.to_log(output)))
            metrics['std_logSNR'].append((np.std(metrics['log_SNR'])))
            metrics['SNR'].append(evaluate.snr((target), (output)))
            metrics['std_SNR'].append((np.std(metrics['SNR'])))

            metrics['zf_nmse'].append(evaluate.nmse(target, Zero_filling))
            metrics['zf_psnr'].append(evaluate.psnr(target, Zero_filling))
            metrics['zf_logSNR'].append(evaluate.snr(evaluate.to_log(target), evaluate.to_log(Zero_filling)))
            metrics['zf_SNR'].append(evaluate.snr((target), (Zero_filling)))
            metrics['zf_ssim'].append(evaluate.ssim(target, Zero_filling))

        self.logger.experiment.add_scalar("Val_logSNR", np.mean(metrics['log_SNR']), self.current_epoch)
        self.logger.experiment.add_scalar("Val_SNR", np.mean(metrics['SNR']), self.current_epoch)
        self.logger.experiment.add_scalar("Val_PSNR", np.mean(metrics['psnr']), self.current_epoch)

        metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        print(metrics, '\n')


        # self.writer.add_scalar("Log_SNR", np.mean(metrics['log_SNR']), self.current_epoch)
        # self.writer.add_scalar("SNR", np.mean(metrics['SNR']), self.current_epoch)
        # self.writer.add_histogram("Histogram of output", output, self.current_epoch)
        # self.writer.add_histogram("Histogram of target", target, self.current_epoch)
        # self.writer.add_histogram("Histogram of Dirty", Zero_filling, self.current_epoch)
        # self.writer.add_scalar("PSNR", np.mean(metrics['psnr']), self.current_epoch)
        # self.writer.add_scalar("SSIM", np.mean(metrics['ssim']), self.current_epoch)

        # self.logger.experiment.add_scalars("Log_SNR", np.mean(metrics['log_SNR']), self.current_epoch)
        # self.logger.experiment.add_histogram("Histogram of output", output, self.current_epoch)
        # self.logger.experiment.add_histogram("Histogram of target", target, self.current_epoch)
        # self.logger.experiment.add_histogram("Histogram of Dirty", Zero_filling, self.current_epoch)
        # self.logger.experiment.add_scalars("PSNR", np.mean(metrics['psnr']), self.current_epoch)
        # self.logger.experiment.add_scalars("SSIM", np.mean(metrics['ssim']), self.current_epoch)

        return dict(log=metrics, **metrics)

    def _visualize(self, val_logs):
        def _normalize(image):
            image = image[np.newaxis]
            image -= image.min()
            return image / image.max()

        def _save_image(image, tag):
            grid = torchvision.utils.make_grid(torch.Tensor(image), nrow=4, pad_value=1)
            self.logger.experiment.add_image(tag, grid)

        # Only process first size to simplify visualization.
        visualize_size = val_logs[0]['output'].shape
        val_logs = [x for x in val_logs if x['output'].shape == visualize_size]
        num_logs = len(val_logs)
        num_viz_images = 16
        step = (num_logs + num_viz_images - 1) // num_viz_images
        outputs, targets = [], []
        for i in range(0, num_logs, step):
            outputs.append(_normalize(val_logs[i]['output'][0]))
            targets.append(_normalize(val_logs[i]['target'][0]))
        outputs = np.stack(outputs)
        targets = np.stack(targets)
        _save_image(targets, 'Target')
        _save_image(outputs, 'Reconstruction')
        _save_image(np.abs(targets - outputs), 'Error')

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Training_loss", avg_loss, self.current_epoch)

    def validation_epoch_end(self, val_logs):
        self._visualize(val_logs)
        return self._evaluate(val_logs)

    def test_epoch_end(self, test_logs):
        t_losses = []
        outputs = defaultdict(list)
        Zero_fills = defaultdict(list)
        orginal = defaultdict(list)

        for log in test_logs:
            t_losses.append(log['test_loss'].cpu().numpy())
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['test_output'][i]))
                Zero_fills[fname].append((slice, log['test_ZF'][i]))
                orginal[fname].append((slice, log['test_target'][i]))

        test_metrics = dict(Test_Loss=t_losses, nmse=[], ssim=[], psnr=[], SNR=[], log_SNR=[], std_nmse=[],
                            std_ssim=[], std_psnr=[], std_logSNR=[], std_SNR=[],
                            zf_nmse=[], zf_ssim=[], zf_psnr=[], zf_logSNR=[], zf_SNR=[])
        for fname in outputs:
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(orginal[fname])])
            Zero_filling = np.stack([zf for _, zf in sorted(Zero_fills[fname])])

            test_metrics['nmse'].append(evaluate.nmse(target, output))
            test_metrics['std_nmse'].append(np.std(test_metrics['nmse']))
            test_metrics['ssim'].append(evaluate.ssim(target, output))
            test_metrics['std_ssim'].append(np.std(test_metrics['ssim']))

            test_metrics['psnr'].append(evaluate.psnr(target, output))
            test_metrics['std_psnr'].append((np.std(test_metrics['psnr'])))
            print(fname,'log_SNR:',evaluate.snr(evaluate.to_log(target), evaluate.to_log(output)),'SNR:',evaluate.snr((target),(output)))
            test_metrics['log_SNR'].append(evaluate.snr(evaluate.to_log(target), evaluate.to_log(output)))

            test_metrics['std_logSNR'].append((np.std(test_metrics['log_SNR'])))
            test_metrics['SNR'].append(evaluate.snr((target), (output)))
            test_metrics['std_SNR'].append((np.std(test_metrics['SNR'])))

            test_metrics['zf_nmse'].append(evaluate.nmse(target, Zero_filling))
            test_metrics['zf_psnr'].append(evaluate.psnr((target), (Zero_filling)))
            test_metrics['zf_logSNR'].append(evaluate.snr(evaluate.to_log(target), evaluate.to_log(Zero_filling)))
            test_metrics['zf_SNR'].append(evaluate.snr((target), (Zero_filling)))
            test_metrics['zf_ssim'].append(evaluate.ssim(target, Zero_filling))
            outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])
            orginal[fname] = np.stack([out for _, out in sorted(orginal[fname])])
            Zero_fills[fname] = np.stack([zf for _, zf in sorted(Zero_fills[fname])])

        # save_ZF(Zero_fills, str(self.hparams.data_path)+"/"+str(self.hparams.scname_test)+"_dirty")
        # save_target(orginal, str(self.hparams.data_path)+"/testset_fullnumpy")

        # save_reconstructions(outputs, str(self.hparams.exp)+str(self.hparams.checkpoint).split('/')[-1].split('-')[0])

        save_reconstructions(outputs, str(self.hparams.data_path)+"/"+str(self.hparams.scname_test)+"_"+str(self.hparams.rec2_ext))

        metrics = {metric: np.mean(values) for metric, values in test_metrics.items()}
        print(metrics, '\n')


        return dict(log=metrics, **metrics)