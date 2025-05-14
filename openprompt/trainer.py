import os, shutil
import sys

sys.path.append(".")

import torch
from torch import nn
from torch.nn.parallel.data_parallel import DataParallel
from openprompt.utils.cuda import model_to_device
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
import dill
import warnings

from typing import Callable, Union, Dict

try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict

from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from openprompt import PromptDataLoader
from openprompt.prompts import *
from openprompt.utils.logging import logger
from openprompt.utils.metrics import classification_metrics, generation_metric
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule
from .utils.sinkhorn_knopp import SinkhornKnopp
from .data_info import gid_ind_ood_index, gid_ood_index


class BaseRunner(object):
    r"""A base runner for training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer,
    or self-training can use other runner class.
    This class is specially implemented for classification.
    For generation task, though it can be integrated in this class
    via `task` option, we keep it as another class for simplicity.

    Args:
        model (:obj:`nn.Module`): One ``nn.Module`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """

    def __init__(self, model: PromptForClassification, config: CfgNode = None,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None, ):
        # self.prototype = None
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.wrap_model()

        self.cur_epoch = 0
        self.best_score = None
        self.global_step = 0
        self.writer = SummaryWriter(os.path.join(self.config.logging.path, 'tensorboard'))
        if not os.path.exists(os.path.join(config.logging.path, 'checkpoints')):
            os.mkdir(os.path.join(config.logging.path, 'checkpoints'))

        self.clean = self.config.train.clean
        self.sk = SinkhornKnopp(num_iters=3, epsilon=0.05)
        self.negative_infinity = torch.finfo(torch.float32).min

        project_dim = 256
        config.hidden_size = 768
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size, bias=False),
                                 nn.BatchNorm1d(config.hidden_size), nn.ReLU(inplace=True),
                                 nn.Linear(config.hidden_size, project_dim, bias=False)).cuda()

        if self.config.train.mode == 'discover':
            # 初始化prototype token
            from .pre_prototype import get_prototype_token
            verb_type = self.config.verbalizer
            self.verb_path = str(getattr(self.config, verb_type).file_path)
            self.prototype_token = get_prototype_token(self.verb_path)
            # 初始化prototype
            self.prototype = self.encode_prototype()
            self.ins_emb = torch.zeros_like(self.prototype)

            data_setup = self.config.dataset.path.split('/')[-1].lower()
            self.gid_index_dict = gid_ind_ood_index[data_setup]
            self.ood_index = gid_ood_index[data_setup]

            # self.gid_index_dict = gid_sd_60_index_dict  # dropout 值输出  # print(self.model.prompt_model.plm.config.hidden_dropout_prob)  # exit()

    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.close()

    def log(self, name, y, x):
        if self.clean: return  # TODO add more types
        self.writer.add_scalar(name, y, x)

    def set_stop_criterion(self):
        """Total training steps, either controlled by num_training_steps or num_epochs"""
        if hasattr(self.config.train, "num_training_steps") and self.config.train.num_training_steps is not None:
            if self.config.train.num_epochs is not None:
                logger.warning("num_training_steps set explicitly, num_epochs is not in use.")
            self.num_training_steps = self.config.train.num_training_steps
            self.num_epochs = int(1e8)  # set to a large number
        else:
            if self.config.train.num_epochs is None:
                raise RuntimeError("At least num_training_steps & num_epochs should be specified.")
            self.num_training_steps = self.steps_per_epoch * self.config.train.num_epochs
            self.num_epochs = self.config.train.num_epochs

    @property
    def steps_per_epoch(self) -> int:
        """num of training steps per epoch"""
        batches = len(self.train_dataloader)
        effective_accum = self.config.train.gradient_accumulation_steps
        return (batches // effective_accum)

    def wrap_model(self):
        self.model = model_to_device(self.model, self.config.environment)

    @property
    def inner_model(self):
        return self.model.module if isinstance(self.model, DataParallel) else self.model

    def configure_optimizers(self):
        r"""config the optimizer and scheduler for

        1. model

        2. template

        3. verbalizer(optional)
        """

        optimizers = []
        schedulers = []

        # print(type(self.inner_model.plm))
        if self.config.plm.optimize.freeze_para:
            for name, param in self.inner_model.plm.named_parameters():
                if "encoder" in name:  # 如果名称中包含 "encoder"
                    if "11" not in name:  # 如果不是第 11 层
                        param.requires_grad = False  # 冻结参数
                    else:
                        param.requires_grad = True  # 解冻第 11 层参数
                else:
                    param.requires_grad = True  # 不冻结参数

        # if not self.config.plm.optimize.freeze_para:
        if not self.config.plm.optimize.freeze_para or 1:
            no_decay = self.config.plm.optimize.no_decay
            weight_decay = self.config.plm.optimize.weight_decay
            optimizer_grouped_parameters = [{'params': [p for n, p in self.inner_model.plm.named_parameters() if
                                                        not any(nd in n for nd in no_decay)],
                                             'weight_decay': weight_decay}, {
                                                'params': [p for n, p in self.inner_model.plm.named_parameters() if
                                                           any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
                                            # todo 添加prototype的参数优化器，暂时放在这里，后续可能还需要改进
                                            {'params': [p for n, p in self.mlp.named_parameters() if
                                                        not any(nd in n for nd in no_decay)],
                                             'weight_decay': weight_decay}, {
                                                'params': [p for n, p in self.mlp.named_parameters() if
                                                           any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

            plm_optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.plm.optimize.lr,
                                  betas=self.config.plm.optimize.betas, eps=self.config.plm.optimize.eps)
            optimizers.append(plm_optimizer)
            if self.config.plm.optimize.scheduler is not None:
                plm_scheduler = get_linear_schedule_with_warmup(plm_optimizer,
                                                                num_warmup_steps=self.config.plm.optimize.scheduler.num_warmup_steps,
                                                                num_training_steps=self.num_training_steps)
                schedulers.append(plm_scheduler)

        class Dummy:
            pass

        template_config = self.config[self.config.template]
        if hasattr(template_config,
                   "optimize") and template_config.optimize is not None:  # TODO should add optimize config in each yaml
            if not hasattr(self.inner_model.template, "optimize"):
                # using default gradient descent optimizer.
                optimizer_grouped_parameters = [{
                    'params': [p for name, p in self.inner_model.template.named_parameters() if
                               'raw_embedding' not in name]}]
                if template_config.optimize.name.lower() == "adamw":
                    template_optimizer = AdamW(optimizer_grouped_parameters, lr=template_config.optimize.lr,
                                               eps=template_config.optimize.adam_epsilon)
                    optimizers.append(template_optimizer)
                    if hasattr(template_config.optimize,
                               "scheduler") and template_config.optimize.scheduler is not None:
                        template_scheduler = get_linear_schedule_with_warmup(template_optimizer,
                                                                             num_warmup_steps=template_config.optimize.scheduler.num_warmup_steps,
                                                                             num_training_steps=self.num_training_steps)
                        schedulers.append(template_scheduler)
                elif template_config.optimize.name.lower() == "adafactor":
                    template_optimizer = Adafactor(optimizer_grouped_parameters, lr=template_config.optimize.lr,
                                                   weight_decay=1e-5, relative_step=False, scale_parameter=False,
                                                   warmup_init=False)
                    # template_scheduler = AdafactorSchedule(template_optimizer)
                    optimizers.append(template_optimizer)  # schedulers.append(template_scheduler)
                else:
                    raise NotImplementedError("Template Optimizer not Implemented!")
            else:
                template_optimizer = Dummy()
                # resemble a pytorch optimizer for unified training.
                setattr(template_optimizer, "step", self.inner_model.template.optimize)
                setattr(template_optimizer, "zero_grad", lambda: None)
                optimizers.append(template_optimizer)

        if hasattr(self.inner_model, "verbalizer") and self.inner_model.verbalizer:
            verbalizer_config = self.config[self.config.verbalizer]
            if hasattr(verbalizer_config, "optimize") and verbalizer_config.optimize is not None:
                if not hasattr(self.inner_model.verbalizer, "optimize"):
                    # using default gradient descent optimizer.
                    verbalizer_optimizer = AdamW(self.inner_model.verbalizer.parameters(),
                                                 lr=verbalizer_config.optimize.lr)
                    optimizers.append(verbalizer_optimizer)
                    if hasattr(verbalizer_config.optimize,
                               "scheduler") and verbalizer_config.optimize.scheduler is not None:
                        verbalizer_scheduler = get_linear_schedule_with_warmup(verbalizer_optimizer,
                                                                               num_warmup_steps=verbalizer_config.optimize.scheduler.num_warmup_steps,
                                                                               num_training_steps=self.num_training_steps)
                        schedulers.append(verbalizer_scheduler)
                else:
                    verbalizer_optimizer = Dummy()
                    # resemble a pytorch optimizer for unified training.
                    setattr(verbalizer_optimizer, "step", self.inner_model.verbalizer.optimize)
                    setattr(verbalizer_optimizer, "zero_grad", lambda: None)
                    optimizers.append(verbalizer_optimizer)

        self.optimizers = optimizers
        self.schedulers = schedulers

    def checkpoint_path(self, ckpt: str) -> str:
        return os.path.join(os.path.join(self.config.logging.path, "checkpoints"), f'{ckpt}.ckpt')

    def load_checkpoint(self, ckpt: str, load_state=True) -> bool:
        logger.info(f"Loading Checkpoint {self.checkpoint_path(ckpt)}...")
        try:
            state_dict = torch.load(self.checkpoint_path(ckpt), pickle_module=dill, map_location="cpu")
        except FileNotFoundError:
            logger.warning(f"Checkpoint {self.checkpoint_path(ckpt)} not found")
            return False

        # load state to model
        self.model = self.inner_model
        self.model.load_state_dict(state_dict['state_dict'])

        if load_state:
            # load state to optimizers
            for optimizer, op_state in zip(self.optimizers, state_dict['optimizer']):
                if isinstance(optimizer, torch.optim.Optimizer):
                    optimizer.load_state_dict(op_state)
            for scheduler, sc_state in zip(self.schedulers, state_dict['scheduler']):
                if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
                    with warnings.catch_warnings(record=True):
                        scheduler.load_state_dict(sc_state)

            # load training state
            self.cur_epoch = state_dict['cur_epoch'] + 1
            self.best_score = state_dict['best_score']
            self.global_step = state_dict['global_step']
        logger.info(f"Load Checkpoint finished, the current validation metric: {state_dict['validation_metric']}")
        return True

    def save_checkpoint(self, ckpt: str, save_state=True, extra: dict = {}, copy: str = None):
        if self.clean: return
        logger.info(f"Saving checkpoint {self.checkpoint_path(ckpt)}...")
        state_dict = {"state_dict": self.inner_model.state_dict(), }
        state_dict.update(extra)

        if save_state:
            state_dict["optimizer"] = [opt.state_dict() if isinstance(opt, torch.optim.Optimizer) else None for opt in
                                       self.optimizers]
            with warnings.catch_warnings(record=True):
                state_dict["scheduler"] = [
                    sch.state_dict() if isinstance(sch, torch.optim.lr_scheduler._LRScheduler) else None for sch in
                    self.schedulers]

            state_dict.update(
                {"cur_epoch": self.cur_epoch, "best_score": self.best_score, "global_step": self.global_step, })
        torch.save(state_dict, self.checkpoint_path(ckpt), pickle_module=dill)
        if copy:
            logger.info(f"Copying checkpoint {self.checkpoint_path(ckpt)} to {self.checkpoint_path(copy)}...")
            shutil.copyfile(self.checkpoint_path(ckpt), self.checkpoint_path(copy))
        logger.info(f"Save Checkpoint finished")

    def save_results(self, split, results: dict):
        if self.clean: return
        for name, values in results.items():
            file_name = os.path.join(self.config.logging.path, f"{split}_{name}.txt")
            with open(file_name, 'w') as fout:
                for value in values:
                    print(value, file=fout)

    def inference_epoch(self, split: str):
        outputs = []
        self.model.eval()
        with torch.no_grad():
            data_loader = self.valid_dataloader if split == 'validation' else self.test_dataloader
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=split)):
                batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()

                outputs.append(self.inference_step(batch, batch_idx))

        metrics = self.inference_epoch_end(split, outputs)
        logger.info(f"{split} Performance: {metrics}")
        for metric_name, metric in metrics.items():
            self.log(f'{split}/{metric_name}', metric, self.cur_epoch)
        return metrics.popitem(last=False)[1]  # TODO the first metric is the most important one

    def training_epoch(self, epoch):
        self.model.train()
        self.model.zero_grad()
        total_loss = 0.0
        sum_loss = 0.0


        with tqdm(total=self.steps_per_epoch, desc=f"train epoch: {epoch}") as pbar:
            for batch_idx, batch in enumerate(self.train_dataloader):
                batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
                loss = self.training_step(batch, batch_idx)

                if self.config.train.gradient_accumulation_steps > 1:
                    loss = loss / self.config.train.gradient_accumulation_steps
                sum_loss += loss.item()
                loss.backward()
                if (batch_idx + 1) % self.config.train.gradient_accumulation_steps == 0:
                    pbar.set_postfix({'loss': sum_loss})
                    self.log('train/loss', sum_loss, self.global_step)
                    # logger.info("{} {}".format(self.inner_model.template.soft_embeds.data.mean().item(),self.global_step))

                    if self.config.train.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.max_grad_norm)

                    for optimizer in self.optimizers:
                        optimizer.step()
                    for scheduler in self.schedulers:
                        scheduler.step()
                    for optimizer in self.optimizers:
                        optimizer.zero_grad()

                    total_loss += sum_loss
                    sum_loss = 0.
                    self.global_step += 1
                    pbar.update(1)
                if self.global_step >= self.num_training_steps:
                    logger.info(
                        f"Training epoch {epoch}, num_steps {self.global_step}, avg_loss: {total_loss / self.steps_per_epoch:.4f}, total_loss: {total_loss:.4f}")
                    return -1  # an indicator of stopping the training
        logger.info(
            f"Training epoch {epoch}, num_steps {self.global_step},  avg_loss: {total_loss / self.steps_per_epoch:.4f}, total_loss: {total_loss:.4f}")
        return 1

    def on_fit_start(self):
        """Some initialization works"""
        pass

    def encode_prototype(self):
        if self.config.train.mode == 'discover':
            with torch.no_grad():
                prototype = self.model(self.prototype_token, input_prototype=True)
                mask = self.prototype_token['attention_mask'].unsqueeze(-1).float()
                masked_prototype = prototype * mask
                seq_lengths = mask.sum(dim=1)
                mean_prototype = masked_prototype.sum(dim=1) / seq_lengths
            return F.normalize(mean_prototype.clone().detach(), dim=-1)  # return mean_prototype.clone().detach()

    def fit(self, ckpt: Optional[str] = None):
        self.set_stop_criterion()
        self.configure_optimizers()

        if ckpt:
            # 重置最佳值，重置当前训练回合
            self.best_score = 0
            self.cur_epoch = 0
            if not self.load_checkpoint(ckpt):
                logger.warning("Train from scratch instead ...")
        if self.cur_epoch == 0:
            self.on_fit_start()

        no_improvement_num = 0
        for self.cur_epoch in range(self.cur_epoch, self.num_epochs):
            continue_training = self.training_epoch(self.cur_epoch)
            score = self.inference_epoch("validation")
            copy = None
            if self.best_score is None or ((score - self.best_score) > 0) == self.config.checkpoint.higher_better:
                copy = 'best'
                self.best_score = score
                no_improvement_num = 0
                if self.config.train.mode == 'pretrain':
                    self.save_checkpoint('best_pretrain', extra={"validation_metric": score})
                elif self.config.train.mode == 'discover':
                    self.save_checkpoint('best_discover', extra={"validation_metric": score})
                elif self.config.train.mode == 'detect':
                    self.save_checkpoint('best_detect', extra={"validation_metric": score})

            else:
                no_improvement_num += 1
                logger.info(f'Epoch {self.cur_epoch}, No improvement num: {no_improvement_num}')
            # self.save_checkpoint('last', extra={"validation_metric": score}, copy = copy)
            if continue_training == -1 or no_improvement_num >= self.config.checkpoint.patience:
                logger.info("Stop training by reaching maximum num_training_steps")
                break
        return self.best_score

    def test(self, ckpt: Optional[str] = None) -> dict:
        if ckpt:
            if not self.load_checkpoint(ckpt, load_state=False):
                logger.error("Test cannot be performed")
                exit()
        return self.inference_epoch("test")

    def run(self, ckpt: Optional[str] = None) -> dict:
        self.fit(ckpt)
        if self.config.train.mode == 'pretrain':
            return self.test(ckpt=None if self.clean else 'best_pretrain')
        elif self.config.train.mode == 'discover':
            return self.test(ckpt=None if self.clean else 'best_discover')
        elif self.config.train.mode == 'detect':
            return self.test(ckpt=None if self.clean else 'best_detect')


class ClassificationRunner(BaseRunner):
    r"""A runner for simple training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer,
    or self-training can use other runner class.
    This class is specially implemented for classification.
    For generation task, though it can be integrated in this class
    via `task` option, we keep it as another class for simplicity.

    Args:
        model (:obj:`PromptForClassification`): One ``PromptForClassification`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """

    def __init__(self, model: PromptForClassification, config: CfgNode = None,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None, loss_function: Optional[Callable] = None,
                 id2label: Optional[Dict] = None, ):
        super().__init__(model=model, config=config, train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader, test_dataloader=test_dataloader, )
        self.loss_function = loss_function if loss_function else self.configure_loss_function()
        self.id2label = id2label
        self.label_path_sep = config.dataset.label_path_sep

    def configure_loss_function(self, ):
        r"""config the loss function if it's not passed."""
        if self.config.classification.loss_function == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif self.config.classification.loss_function == "nll_loss":
            return torch.nn.NLLLoss()
        else:
            raise NotImplementedError

    def inference_step(self, batch, batch_idx):
        label = batch.pop('label')
        logits, _ = self.model(batch)
        # 添加屏蔽操作
        if self.config.train.mode == 'detect':
            pred = torch.argmax(logits, dim=-1)
            return pred.cpu().tolist(), label.cpu().tolist()

        pred = torch.argmax(logits, dim=-1)
        return pred.cpu().tolist(), label.cpu().tolist()

    def inference_epoch_end(self, split, outputs):
        preds = []
        labels = []
        for pred, label in outputs:
            preds.extend(pred)
            labels.extend(label)

        self.save_results(split, {'preds': preds, 'labels': labels, })

        from sklearn.metrics import accuracy_score, f1_score
        print(accuracy_score(preds, labels))
        # print(f1_score(preds, labels, average='micro'))
        metrics = OrderedDict()
        for metric_name in self.config.classification.metric:
            metric = classification_metrics(preds, labels, metric_name, id2label=self.id2label,
                                            label_path_sep=self.label_path_sep)
            metrics[metric_name] = float(round(metric, 4))
        return metrics

    @staticmethod
    def cross_entropy_loss(logits, one_hot_labels):
        probs = F.softmax(logits, dim=1)
        loss = -(one_hot_labels * torch.log(probs + 1e-8)).sum(dim=1).mean()
        return loss

    @staticmethod
    def normalize_vector(vec):
        """Normalize a vector to have unit length."""
        norm = torch.norm(vec, p=2)
        if norm == 0:
            return vec
        return vec / norm

    def training_step(self, batch, batch_idx):


        if self.config.train.mode == 'pretrain' or self.config.train.mode == 'detect':
            logits, _ = self.model(batch)
            loss_ce = self.loss_function(logits, batch['label'])
            return loss_ce


        # sl等于监督学习，ml等于度量学习，有两个view
        sl_logits1, raw_emb1 = self.model(batch)
        sl_logits2, raw_emb2 = self.model(batch)


        mask = batch['attention_mask']
        lenths = mask.sum(dim=1, keepdim=True).float()
        masked_emb1 = raw_emb1 * mask.unsqueeze(-1)
        masked_emb2 = raw_emb2 * mask.unsqueeze(-1)
        emb1 = masked_emb1.sum(dim=1) / lenths
        emb2 = masked_emb2.sum(dim=1) / lenths


        ml_logits1 = F.cosine_similarity(F.normalize(emb1.unsqueeze(1), dim=-1), self.prototype.unsqueeze(0), dim=-1)
        ml_logits2 = F.cosine_similarity(F.normalize(emb2.unsqueeze(1), dim=-1), self.prototype.unsqueeze(0), dim=-1)

        bsz = len(batch['label'])
        from .utils.instance_loss import InstanceLoss
        criterion = InstanceLoss(bsz, 0.5, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # loss_ins = criterion(emb1, emb2)
        loss_ins = criterion(F.normalize(self.mlp(emb1), dim=-1), F.normalize(self.mlp(emb2), dim=-1))
        # loss_ins = 0

        # dc指的是数据一致性损失，lc指的是标签一致性损失
        kl_divergence = nn.KLDivLoss(reduction='mean')
        loss_dc = (kl_divergence(F.log_softmax(emb1, dim=-1), F.softmax(emb2, dim=-1)) + kl_divergence(
            F.log_softmax(emb2, dim=-1), F.softmax(emb1, dim=-1))) * 1 / 2

        # self.config.dataset.path.split('/')[-1].lower()

        target_view1 = torch.zeros_like(sl_logits1)
        target_view2 = torch.zeros_like(sl_logits1)

        index_ood = []
        for idx, label in enumerate(batch['label']):
            if label.item() in self.gid_index_dict['ind']:
                # print("if")
                # exit()
                target_view1[idx] = F.one_hot(label, num_classes=self.model.verbalizer.num_classes).float().cuda()
                target_view2[idx] = F.one_hot(label, num_classes=self.model.verbalizer.num_classes).float().cuda()
            else:
                index_ood.append(idx)

        ind_list = self.gid_index_dict['ind'] + self.ood_index
        with torch.no_grad():
            ood_logit1 = sl_logits1[index_ood, :].clone().detach()
            ood_logit2 = sl_logits2[index_ood, :].clone().detach()
            # todo 尝试下屏蔽时使用0还是负无穷，或者直接不屏蔽
            ## 使用0或负无穷屏蔽效果一致，不能不屏蔽，loss会nan
            ood_logit1[:, ind_list] = self.negative_infinity
            ood_logit2[:, ind_list] = self.negative_infinity
            # ood_logit1[:, ind_list] = 0
            # ood_logit2[:, ind_list] = 0
            # todo 探究一下使用softmax对结果是否有影响
            target_view1[index_ood] = self.sk(F.softmax(ood_logit1, dim=-1))
            target_view2[index_ood] = self.sk(F.softmax(ood_logit2,
                                                        dim=-1))  # target_view1[index_ood] = self.sk(ood_logit1)  # target_view2[index_ood] = self.sk(ood_logit2)

        self.prototype = self.encode_prototype()

        loss_sim_1 = self.cross_entropy_loss(ml_logits1, target_view2)
        loss_sim_2 = self.cross_entropy_loss(ml_logits2, target_view1)
        loss_sim = (loss_sim_1 + loss_sim_2) / 2


        kl_divergence = nn.KLDivLoss(reduction='mean')
        loss_lc = (kl_divergence(F.log_softmax(ml_logits1, dim=-1), F.softmax(sl_logits2, dim=-1)) + kl_divergence(
            F.log_softmax(sl_logits1, dim=-1), F.softmax(ml_logits2, dim=-1))) * 1 / 2

        # todo 这里需要设置个proto_targets，对于ind部分使用原生label id，对于ood部分使用相似logits然后one hot化。
        proto_target_view1 = torch.zeros_like(sl_logits1)
        proto_target_view2 = torch.zeros_like(sl_logits1)
        for idx, label in enumerate(batch['label']):
            if label.item() in self.gid_index_dict['ind']:
                proto_target_view1[idx] = F.one_hot(label, num_classes=self.model.verbalizer.num_classes).float().cuda()
                proto_target_view2[idx] = F.one_hot(label, num_classes=self.model.verbalizer.num_classes).float().cuda()
            else:
                # todo 屏蔽掉ind部分的logits
                negative_infinity = torch.finfo(torch.float32).min
                with torch.no_grad():
                    new_piece_logit1 = ml_logits1[idx, :].clone()
                    new_piece_logit2 = ml_logits2[idx, :].clone()
                    new_piece_logit1[self.gid_index_dict['ind'] + self.ood_index] = negative_infinity
                    new_piece_logit2[self.gid_index_dict['ind'] + self.ood_index] = negative_infinity
                    proto_target_view1[idx] = F.one_hot(new_piece_logit1.argmax(),
                                                        num_classes=self.model.verbalizer.num_classes).float().cuda()
                    proto_target_view2[idx] = F.one_hot(new_piece_logit2.argmax(),
                                                        num_classes=self.model.verbalizer.num_classes).float().cuda()

        loss_ce_1 = self.cross_entropy_loss(sl_logits1, proto_target_view2)
        loss_ce_2 = self.cross_entropy_loss(sl_logits2, proto_target_view1)
        loss_ce = (loss_ce_1 + loss_ce_2) / 2


        loss_cr = loss_dc + loss_lc
        loss_cp = loss_ce + loss_sim
        loss_cl = loss_ins


        loss = loss_cr + loss_cp + loss_cl
        return loss

    def on_fit_start(self):
        """Some initialization works"""
        self.prompt_initialize()

    def prompt_initialize(self):
        verbalizer_config = self.config[self.config.verbalizer]
        template_config = self.config[self.config.template]
        if not hasattr(self.inner_model.verbalizer, "optimize_to_initialize") and not hasattr(self.inner_model.template,
                                                                                              "optimize_to_initialize"):
            return None
        if hasattr(verbalizer_config, "init_using_split"):
            using_split = verbalizer_config.init_using_split
        elif hasattr(template_config, "init_using_split"):
            using_split = template_config.init_using_split
        else:
            using_split = "valid"

        if using_split == "train":
            dataloader = self.train_dataloader
        elif using_split == "valid":
            dataloader = self.valid_dataloader
        else:
            raise NotImplementedError

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Init_using_{}".format(using_split)):
                batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
                logits, _ = self.model(batch)
            if hasattr(self.inner_model.verbalizer, "optimize_to_initialize"):
                self.inner_model.verbalizer.optimize_to_initialize()
            if hasattr(self.inner_model.template, "optimize_to_initialize"):
                self.inner_model.template.optimize_to_initialize()

        self.wrap_model()


class GenerationRunner(BaseRunner):
    r"""A runner for simple training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer,
    or self-training can use other runner class.
    This class is specially implemented for generation.

    Args:
        model (:obj:`PromptForGeneration`): One ``PromptForGeneration`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
    """

    def __init__(self, model: PromptForGeneration, config: CfgNode = None,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None, ):
        super().__init__(model=model, config=config, train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader, test_dataloader=test_dataloader, )

    def inference_step(self, batch, batch_idx):
        target = batch['tgt_text']  # TODO pop?
        _, pred = self.model.generate(batch, **self.config.generation)
        return pred, target  # these are already a cpu list

    def inference_epoch_end(self, split, outputs):
        preds = []
        targets = []
        for pred, target in outputs:
            preds.extend(pred)
            targets.extend(target)

        self.save_results(split, {'preds': preds, 'targets': targets})

        metrics = OrderedDict()
        for metric_name in self.config.generation.metric:
            metric = generation_metric(preds, targets, metric_name)
            metrics[metric_name] = metric
        return metrics

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        return loss
