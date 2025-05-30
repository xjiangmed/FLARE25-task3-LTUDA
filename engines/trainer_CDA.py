import os
import time
import torch
from loguru import logger
from tqdm import tqdm
from utils.common_function import to_cuda
from .metrics import AverageMeter,run_online_evaluation
import torch.distributed as dist
from torch.nn.functional import one_hot
import wandb
import shutil
from copy import deepcopy
from .utils_CDA import obtain_bbox, mix


class ModelEMA(object):
    def __init__(self, model, decay):
        self.ema = deepcopy(model)
        self.ema.cuda()
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)
            p.detach_()

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            
            for k in self.param_keys:
                if self.ema_has_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if self.ema_has_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

    def get_model(self):
        return self.ema

class Trainer_CDA:
    def __init__(self, config, labeled_train_loader, unlabeled_train_loader, val_loader, model,losses,optimizer,lr_scheduler):
        self.config = config
       
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.losses = losses
        self.model = model # student_model
        
        # 添加 DDP 设置
        if self.config.DIS:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self._get_rank()],
                output_device=self._get_rank(),
                find_unused_parameters=True
            )
            self.model._set_static_graph()
        
        # 修改 EMA 初始化
        self.ema_model = ModelEMA(self.model.module if hasattr(self.model, 'module') else self.model, 0.999)
        self.teacher_model = self.ema_model.get_model()
        self.teacher_model.eval()
            
        self.labeled_train_loader = labeled_train_loader
        self.unlabeled_train_loader = unlabeled_train_loader
        self.val_loader = val_loader
        self.batch_size = config.DATALOADER.BATCH_SIZE
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_steps = len(self.unlabeled_train_loader)
        self.start_epoch = 1
        if self._get_rank() == 0:
            if config.RESUME:
                self.start_epoch = self._load_checkpoint(is_fintune=False)
            if config.FINETUNE:
                print(
                    "FINETUNE: checkpoint path: {}".format(
                        os.path.join(
                            self.config.FINE_MODEL_PATH, self.config.CHECK_POINT_NAME
                        )
                    )
                )
                time.sleep(5)
                self._load_checkpoint(is_fintune=True)
                self.checkpoint_dir = self.config.FINE_MODEL_PATH
            else:
                self.checkpoint_dir = os.path.join(config.SAVE_DIR, config.EXPERIMENT_ID)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.CDA_cutmix = True
            
    def train(self):
        
        for epoch in range(self.start_epoch, self.config.TRAIN.EPOCHS+1):
            if self.config.DIS:
                self.labeled_train_loader.sampler.set_epoch(epoch)
                self.unlabeled_train_loader.sampler.set_epoch(epoch)
                
            self._train_epoch(epoch)
            if self.val_loader is not None and epoch % self.config.TRAIN.VAL_NUM_EPOCHS == 0:
                results = self._valid_epoch(epoch)
                if self._get_rank()==0 :
                    logger.info(f'## Info for epoch {epoch} ## ')
                    for k, v in results.items():
                        logger.info(f'{str(k):15s}: {v}')
            if epoch % self.config.TRAIN.SAVE_PERIOD == 0 and self._get_rank()==0:
            # if epoch % 5 == 0 and self._get_rank()==0:
                self._save_checkpoint(epoch)
                self._save_checkpoint_teacher(epoch)

    
    def _train_epoch(self, epoch):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.DICE = AverageMeter()
    
        self.model.train()
        self.teacher_model.eval()
        
        tbar = tqdm(self.unlabeled_train_loader, ncols=150)
        labeled_dataloader_iter = iter(self.labeled_train_loader)
        tic = time.time()

        for idx, (data_ori_ul, flag_ul, data_infos_ul) in enumerate(tbar):
            try:
                data_ori_l, flag_l, data_infos_l = labeled_dataloader_iter.next()
            except StopIteration:
                labeled_dataloader_iter = iter(self.labeled_train_loader)
                data_ori_l, flag_l, data_infos_l = labeled_dataloader_iter.next()
            
            self.data_time.update(time.time() - tic)
            
            # 准备数据，使用克隆版本
            img_l = to_cuda(data_ori_l["data"]).clone()
            gt_l = [to_cuda(seg).clone() for seg in data_ori_l["seg"]]
            img_ul = to_cuda(data_ori_ul["data"]).clone()
            
            if img_l.shape[0] != self.batch_size or img_ul.shape[0] != self.batch_size:
                continue

            # 使用 teacher_model 生成伪标签
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config.AMP):
                outputs_unlabeled = self.teacher_model(img_ul, deepsup=False)
                outputs_unlabeled_soft = torch.softmax(outputs_unlabeled, dim=1)
                pseudo_labels = torch.argmax(outputs_unlabeled_soft, dim=1, keepdim=True).clone()
            
            # 准备增强数据
            dwh_shape = img_l.shape[-3:]
            bsl = img_l.shape[0]
            bsul = img_ul.shape[0]
            target_l = gt_l[0].clone()
            
            # 确保所有张量都在同一个设备上，使用克隆版本
            MixMask = obtain_bbox(bsl+bsul, dwh_shape).cuda().unsqueeze(1).clone()
            
            # 执行数据增强，确保使用克隆版本
            if self.CDA_cutmix:
                # 克隆输入张量
                img_l_clone = img_l.clone()
                img_ul_clone = img_ul.clone()
                target_l_clone = target_l.clone()
                pseudo_labels_clone = pseudo_labels.clone()
                
                # 执行 mix 操作，确保返回新的张量而不是修改原张量
                img_aug1, rand_index1 = mix(MixMask[:bsl].clone(), img_l_clone, img_ul_clone)
                img_aug2, rand_index2 = mix(MixMask[bsl:bsl+bsul].clone(), img_ul_clone, img_l_clone)
                pseudo_labels_aug1, _ = mix(MixMask[:bsl].clone(), target_l_clone, pseudo_labels_clone, rand_index1)
                pseudo_labels_aug2, _ = mix(MixMask[bsl:bsl+bsul].clone(), pseudo_labels_clone, target_l_clone, rand_index2)
            
            # 合并批次，使用克隆版本
            volume_batch = torch.cat([img_aug1.clone(), img_aug2.clone()], 0)
            label_batch = torch.cat([pseudo_labels_aug1.clone(), pseudo_labels_aug2.clone()], 0)
            
            # 清除之前的梯度
            self.optimizer.zero_grad(set_to_none=True)
            
            # 单次前向传播和反向传播
            with torch.cuda.amp.autocast(enabled=self.config.AMP):
                # 将两个输入拼接在一起
                combined_input = torch.cat([img_l, volume_batch], dim=0)
                # 一次前向传播
                combined_output = self.model(combined_input, deepsup=True)
                
                # 分开使用结果，处理多尺度输出
                outputs_l = [out[:bsl] for out in combined_output]  # 有标签数据的多尺度输出
                outputs_ul = [out[bsl:] for out in combined_output]  # 无标签数据的多尺度输出
                
                # 计算损失
                labeled_loss = self.losses['ct'](outputs_l, gt_l)
                pseudo_loss_aug1 = self.losses['mr'](outputs_ul[0][:bsl], label_batch[:bsl])  # 只使用第一个尺度的输出
                pseudo_loss_aug2 = self.losses['mr'](outputs_ul[0][bsl:2*bsl], label_batch[bsl:2*bsl])  # 只使用第一个尺度的输出
                
                # 总损失
                loss = labeled_loss + pseudo_loss_aug1 + pseudo_loss_aug2

            # 反向传播
            if self.config.AMP:
                self.scaler.scale(loss).backward()
                if self.config.TRAIN.DO_BACKPROP:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.TRAIN.DO_BACKPROP:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.optimizer.step()
            
            # 更新 teacher model (在优化器步骤之后)
            with torch.no_grad():
                # 确保模型处于评估模式
                self.model.eval()
                # 更新 EMA
                self.ema_model.update(self.model)
                # 获取更新后的 teacher model
                self.teacher_model = self.ema_model.get_model()
                self.teacher_model.eval()
                # 恢复训练模式
                self.model.train()

            # 更新指标
            self.total_loss.update(loss.item())
            self.batch_time.update(time.time() - tic)
            self.DICE.update(run_online_evaluation(outputs_l, gt_l[0]))
            
            tbar.set_description(
                'TRAIN ({}) | Loss: {} | Labeled DICE {} |B {} D {} |'.format(
                    epoch, self.total_loss.average, self.DICE.average, self.batch_time.average, self.data_time.average))
            tic = time.time()
       
            self.lr_scheduler.step_update(epoch * self.num_steps + idx)
            
        if self._get_rank()==0:
            wandb.log({'train/loss': self.total_loss.average,
                    'train/dice': self.DICE.average,
                    'train/lr': self.optimizer.param_groups[0]['lr']},
                      step=epoch)
          
    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.DICE = AverageMeter()

        self.model.eval()

        tbar = tqdm(self.val_loader, ncols=150)
        tic = time.time()
        with torch.no_grad():

            for idx, data in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = to_cuda(data["data"])
                gt = to_cuda(data["seg"])
                
                with torch.cuda.amp.autocast(enabled=self.config.AMP):
                    
                    pre = self.model(img)
                    loss = self.loss(pre, gt)

                self.total_loss.update(loss.item())
                self.batch_time.update(time.time() - tic)
                
                self.DICE.update(run_online_evaluation(pre, gt))
                tbar.set_description(
                'TEST ({}) | Loss: {} | DICE {} |B {} D {} |'.format(
                    epoch, self.total_loss.average, self.DICE.average, self.batch_time.average, self.data_time.average))
                tic = time.time()
        if self._get_rank()==0:        
            wandb.log({'val/loss': self.total_loss.average,
                      'val/dice': self.DICE.average,
                      'val/batch_time': self.batch_time.average,
                      'val/data_time':  self.data_time.average
                      },
                      step=epoch)
        log = {'val_loss': self.total_loss.average,
               'val_dice': self.DICE.average
        }
        return log
    def _get_rank(self):
        """get gpu id in distribution training."""
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    def _save_checkpoint(self, epoch):
        state = {
            "arch": type(self.model).__name__,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }

        filename = os.path.join(self.checkpoint_dir, "{}_checkpoint.pth".format(epoch))
        logger.info(f"Saving a checkpoint: {filename} ...")
        torch.save(state, filename)
        shutil.copy(
            os.path.join(self.checkpoint_dir, "{}_checkpoint.pth".format(epoch)),
            os.path.join(self.checkpoint_dir, "final_checkpoint.pth")
        )
        return filename
    
    def _save_checkpoint_teacher(self, epoch):
        state = {
            "arch": type(self.model).__name__,
            "epoch": epoch,
            "state_dict": self.teacher_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }

        filename = os.path.join(self.checkpoint_dir, "{}_checkpoint_ema.pth".format(epoch))
        logger.info(f"Saving a teacher model checkpoint: {filename} ...")
        torch.save(state, filename)
        shutil.copy(
            os.path.join(self.checkpoint_dir, "{}_checkpoint_ema.pth".format(epoch)),
            os.path.join(self.checkpoint_dir, "final_checkpoint_ema.pth")
        )
        return filename
    
    def _load_checkpoint(self, is_fintune=False):
        checkpoint_path = os.path.join(
            self.config.FINE_MODEL_PATH, self.config.CHECK_POINT_NAME
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

        checkpoint = torch.load(checkpoint_path)

        # load model checkpoint
        self.model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded model from checkpoint at '{checkpoint_path}'")
        # load optimizer parameters
        if self.optimizer is not None and not is_fintune:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"Loaded optimizer state from checkpoint at '{checkpoint_path}'")
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}")

        return start_epoch
    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.DICE = AverageMeter()

    def _print_model_params(self, model):
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}, is_leaf={param.is_leaf}")
