import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from easydict import EasyDict as edict

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import torch
from torch import autocast

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.utilities import grad_norm

from utils.opt import set_optimizer
from utils.sch import set_scheduler
from utils.metrics import NSD, Dice, Haus
from utils.visualzation import VisVolLab
from utils.system import set_gpu, set_seed
from utils.definitions import *

from data.dataloader import NSWDataLoader
from data.registry import data_registry

from model.registry import model_registry
from model.no_more_sw.blocks.custom_callbacks import ReducingTau, StopAtEpoch, EarlyStoppingWithWarmup
import gc


class Trainer(pl.LightningModule):
    def __init__(self, cfg):
        super(Trainer, self).__init__()

        self.cfg = cfg
        self.save_hyperparameters(logger=True)
        self.visualizer = VisVolLab(num_classes=self.cfg.num_classes)

        print(f"Preparing model: {self.cfg.model.name} ...")
        self.net = model_registry[self.cfg.model.name](
            **self.cfg.model,
            input_shape=cfg.input_shape,
            output_shape=cfg.output_shape,
        )

        self.metrics = self.formulate_metric(self.net)
        self.post_process_logit = identity

        torch.cuda.empty_cache()
        gc.collect()

    def on_train_start(self):
        self.logger.log_hyperparams(self.cfg)

    def training_step(self, input_d, _):
        output_d = self.common_step(TRAIN, input_d)
        if output_d is None:
            return None
        return output_d[TOTAL_LOSS]

    def validation_step(self, input_d, batch_idx):
        output_d = self.common_step(VALID, input_d)
        self.output_d = output_d  # for visualization

    def test_step(self, input_d, batch_idx):
        if self.cfg.vis_test:
            self.output_d = self.common_step(TEST, input_d)
            output_d = valmap(
                lambda v: v.detach().cpu() if isinstance(v, torch.Tensor) else v,
                self.output_d,
            )
            vis_function = getattr(self.net, f"visualize_{TEST.lower()}")
            vis_output_d = vis_function(output_d)
            for img_name, img in vis_output_d.items():
                self.log_image_at_current_step(f"{TEST}/{img_name}", img, TEST, batch_idx)

            del output_d, vis_output_d, self.output_d
        else:
            self.common_step(TEST, input_d)

    def common_step(self, mode, input_d):
        if (mode == TEST) and self.cfg.memory:
            torch.cuda.reset_peak_memory_stats()
            self.my_log(f"{TEST}/mem_before", torch.cuda.memory_allocated() / 1024**2, on_step=True)

        with autocast(device_type=self.device.type):
            otuput_d = getattr(self.net, f"{mode.lower()}_step")(input_d)
            if otuput_d is None:
                return otuput_d

        if (mode == TEST) and self.cfg.memory:
            self.my_log(f"{TEST}/mem_after", torch.cuda.memory_allocated() / 1024**2, on_step=True)
            self.my_log(f"{TEST}/mem_peak", torch.cuda.max_memory_allocated() / 1024**2, on_step=True)

        otuput_d = {
            k: v.float() if isinstance(v, torch.Tensor) else v
            for k, v in otuput_d.items()
        }

        loss_d = keyfilter(lambda k: LOSS in k, otuput_d)
        for loss_name, loss_value in loss_d.items():
            self.my_log(f"{mode}/{loss_name}", loss_value)

        if (mode != TRAIN) & (not self.cfg.vis_test) & (self.cfg.test_metric is not None):
            for pred_type, metric_dict in self.metrics[mode].items():
                for _, metric_fn in metric_dict.items():
                    if mode == TEST:
                        processed_logit = self.post_process_logit(
                            otuput_d[pred_type + LOGIT]
                        )
                        metric_fn.update(processed_logit, otuput_d[pred_type + LAB])
                    else:
                        metric_fn.update(
                            otuput_d[pred_type + LOGIT],
                            otuput_d[pred_type + LAB],
                        )

        if (mode == TEST) & (not self.cfg.vis_test):
            del otuput_d
            torch.cuda.empty_cache()
            gc.collect()
        else:
            return otuput_d

    def common_epoch_end(self, mode):

        if (mode != TRAIN) & (not self.cfg.vis_test) & (self.cfg.test_metric is not None):
            for pred_type, metric_dict in self.metrics[mode].items():
                for metric_name, metric_fn in metric_dict.items():
                    for organ_name, metric_val in metric_fn.compute().items():
                        # ugly, think of a better way
                        if pred_type == "":
                            pred_type = "full"
                        else:
                            pred_type = pred_type.rstrip("_")
                        self.log(
                            f"{mode}/{pred_type}/{metric_name}/{organ_name}", metric_val
                        )
                    metric_fn.reset()

    def on_train_epoch_end(self):

        self.common_epoch_end(TRAIN)

        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

        try:
            self.my_log(f"{TRAIN}/tau", self.net.tau)
        except AttributeError:
            pass

    def on_validation_epoch_end(self):

        self.common_epoch_end(VALID)
        output_d = valmap(
            lambda v: v.detach().cpu() if isinstance(v, torch.Tensor) else v,
            self.output_d,
        )
        vis_function = getattr(self.net, f"visualize_{VALID.lower()}")
        vis_output_d = vis_function(output_d)
        for img_name, img in vis_output_d.items():
            self.log_image_at_current_step(f"{VALID}/{img_name}", img)

        try:
            self.logger.experiment.add_histogram(
                "score", output_d["score"].detach().cpu(), self.current_epoch
            )
        except:
            pass
        del output_d, vis_output_d, self.output_d

    def on_test_epoch_end(self):
        self.common_epoch_end(TEST)

    def on_train_epoch_start(self):
        torch.cuda.empty_cache()
        gc.collect()

    def on_validation_epoch_start(self):
        torch.cuda.empty_cache()
        gc.collect()

    def configure_optimizers(self):
        model_optimizer = set_optimizer(self.net.parameters(), opt_kwargs=self.cfg.optimizer)
        model_scheduler = set_scheduler(model_optimizer, sch_kwargs=self.cfg.scheculer)
        return [model_optimizer], [model_scheduler]

    def on_before_optimizer_step(self, optimizer):
        track_grad_modules = [
            "net", 
            "net.global_backbone", "net.global_backbone.encoder", "net.global_backbone.decoder", "net.global_backbone.bottleneck",
            "net.local_backbone", "net.local_backbone.encoder", "net.local_backbone.decoder", "net.local_backbone.bottleneck",
        ]
        for name, module in self.named_modules():
            if len(list(module.parameters())) > 0 and name in track_grad_modules:
                self.my_log(f"grad_norm/{name}", grad_norm(module, norm_type=2)["grad_2.0_norm_total"])

    def formulate_metric(self, net):
        if self.cfg.vis_test or (self.cfg.test_metric == None):
            return

        _dice_mean_metric = lambda do_crop_based: Dice(
            num_classes=self.cfg.num_classes,
            average="mean",
            do_crop_based=do_crop_based,
        )
        _dice_class_metric = lambda do_crop_based: Dice(
            num_classes=self.cfg.num_classes,
            average="none",
            label_keys=self.cfg.label_keys,
            do_crop_based=do_crop_based,
        )

        _nsd_mean_metric = lambda: NSD(
            num_classes=self.cfg.num_classes,
            average="mean",
        )
        _nsd_class_metric = lambda: NSD(
            num_classes=self.cfg.num_classes,
            average="none",
            label_keys=self.cfg.label_keys,
        )

        metrics = {}
        for mode in [TRAIN, VALID, TEST]:
            metrics[mode] = {}
            keys = getattr(net, f"{mode.lower()}_keys")
            for key in keys:
                if mode != TEST:
                    metrics[mode][key] = (
                        {
                            "dice_mean": _dice_mean_metric(False),
                            "dice_class": _dice_class_metric(False),
                        }
                    )
                else:
                    if self.cfg.test_metric == "dsc":
                        metrics[mode][key] = {
                            "dice_mean": _dice_mean_metric(False),
                            "dice_class": _dice_class_metric(False),
                        }
                    elif self.cfg.test_metric == "nsd":
                        metrics[mode][key] = {
                            "nsd_mean": _nsd_mean_metric(),
                            "nsd_class": _nsd_class_metric(),
                        }
        return metrics

    def log_image_at_current_step(self, txt, x, mode = VALID, batch_idx = None):
        if mode == VALID:
            return self.logger.experiment.add_image(
                txt, x.permute(-1, 0, 1), self.current_epoch
            )
        else:
            return self.logger.experiment.add_image(
                txt, x.permute(-1, 0, 1), batch_idx
            )

    def my_log(self, name, val, on_step=False):
        return self.log(name, val, on_step=on_step, on_epoch=not on_step, logger=True)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    cfg = edict(OmegaConf.to_container(cfg, resolve=True))

    # load args and set states
    ###################################################
    set_seed(cfg.seed_number)
    if cfg.accelerator != "tpu":
        set_gpu(cfg.gpu_id)

    # Load Train & Test datasets
    ###################################################
    Dataset = data_registry[cfg.dataset_name]

    if cfg.new_label_rule != None:
        Dataset.new_label_rule = cfg.new_label_rule

    train_dataset = Dataset(
        mode=TRAIN,
        rand_aug_type=cfg.rand_aug_type,
        root_dir="/kaggle/input" if cfg.kaggle else None,
        cache_dir_name=f"{cfg.dataset_name.lower()}-{TRAIN}" if cfg.kaggle else None,
        json_dir="/kaggle/input/brats-json/dataset.json" if cfg.kaggle else None,
    )

    train_loader = NSWDataLoader(
        dataset=train_dataset,
        iteration_per_epoch=cfg.train_iteration_per_epoch,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
    )

    valid_dataset = Dataset(
        mode=VALID,
        rand_aug_type=cfg.rand_aug_type,
        root_dir="/kaggle/input" if cfg.kaggle else None,
        cache_dir_name=f"{cfg.dataset_name.lower()}-{VALID}" if cfg.kaggle else None,
        json_dir="/kaggle/input/brats-json/dataset.json" if cfg.kaggle else None,
    )
    valid_loader = NSWDataLoader(
        dataset=valid_dataset,
        iteration_per_epoch=cfg.val_iteration_per_epoch,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
    )

    test_dataset = Dataset(
        mode=TEST,
        rand_aug_type="none",  # placeholder, not used in test.
        root_dir="/kaggle/input" if cfg.kaggle else None,
        cache_dir_name=f"{cfg.dataset_name.lower()}-{TEST}" if cfg.kaggle else None,
        json_dir="/kaggle/input/brats-json/dataset.json" if cfg.kaggle else None,
    )

    test_loader = NSWDataLoader(
        dataset=test_dataset,
        iteration_per_epoch=None,  # We use the full test dataset
        num_workers=cfg.num_workers,
        batch_size=1,
    )

    # attach extra arguments from dataclass
    ###################################################
    cfg.label_keys = train_dataset.labels
    cfg.num_classes = train_dataset.num_classes
    cfg.num_channel = train_dataset.num_channel
    cfg.down_size_rate = train_dataset.global_downsize_rate
    cfg.patch_size = train_dataset.local_roi_size
    cfg.input_shape = [cfg.num_channel] + train_dataset.global_roi_size
    cfg.output_shape = [cfg.num_classes] + train_dataset.global_roi_size

    cfg.model.down_size_rate = train_dataset.global_downsize_rate
    cfg.model.patch_size = train_dataset.local_roi_size

    # tensorboard Logger
    ###################################################

    # callbacks
    ###################################################
    if cfg.model.name == "GlobalSeg3D":
        ckpt_name = (
            f"{cfg.dataset_name}_{cfg.model.name}_{cfg.model.global_backbone_name}"
        )
    else:
        ckpt_name = (
            f"{cfg.dataset_name}_{cfg.model.name}_{cfg.model.local_backbone_name}"
        )

    if cfg.model.name == "GlobalSeg3D":
        log_file_name = (
            f"{cfg.dataset_name}_{cfg.model.name}_{cfg.model.global_backbone_name}"
        )
    else:
        log_file_name = (
            f"{cfg.dataset_name}_{cfg.model.name}_{cfg.model.local_backbone_name}"
        )

    log_save_dir = f"{cfg.log_base_path}"
    logger = TensorBoardLogger(log_save_dir, log_file_name, 0, default_hp_metric=True)

    ckpt_callback1 = ModelCheckpoint(
        dirpath=f"{cfg.ckpt_base_path}",
        filename=ckpt_name,
        monitor=(f"{VALID}/{TOTAL_LOSS}"),
        save_top_k=cfg.save_top_k,
        save_last=True,
        verbose=True,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor, ckpt_callback1]

    # ugly but lazy to change
    if cfg.model.name == "NSWNet3D":
        tau_reduction = ReducingTau(
            starting_tau=cfg.model.starting_tau,
            final_tau=cfg.model.final_tau,
            reduction_mutiplier=cfg.model.reduction_mutiplier,
        )
        callbacks += [tau_reduction]

    if cfg.early_stopping:
        early_stopping = EarlyStoppingWithWarmup(
            monitor=f"{VALID}/{TOTAL_LOSS}",
            patience=10,
            verbose=True,
            mode="min",
            warmup=cfg.model.reduction_mutiplier * cfg.epoch if cfg.model.name == "NSWNet3D" else 0,
        )
        callbacks += [early_stopping]

    if cfg.stop_at_epoch is not None:
        stop_at_epoch = StopAtEpoch(stop_epoch=cfg.stop_at_epoch)
        callbacks += [stop_at_epoch]

    # profiler
    ###################################################
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=5),
        emit_nvtx=True,
    )

    # init model and trainer
    ###################################################
    print("==== CONFIG ====")
    print(cfg)
    print("==== END CONFIG ====")
    model = Trainer(cfg)
    print("==== MODEL ====")
    print(model)
    print("==== END MODEL ====")

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        profiler=profiler if cfg.profile_debug else None,
        precision=cfg.precision,
        max_epochs=cfg.epoch,
        logger=logger,
        callbacks=callbacks,
        strategy=cfg.strategy,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        gradient_clip_val=1,
        devices=1,
        limit_train_batches=cfg.train_iteration_per_epoch,
    )


    # train, valid, and test
    ###################################################
    if not cfg.test_only:
        trainer.fit(
            model,
            train_loader,
            valid_loader,
            ckpt_path=cfg.ckpt_path,
        )
    else:
        trainer.test(
            model,
            [test_loader],
            ckpt_path=cfg.ckpt_path,
        )

if __name__ == "__main__":

    main()
