from .dataset import ClassifcationDatasetMultiCrop
from .dataset import ClassifcationDatasetMultiCropMultiHead
import collections
from catalyst.dl.runner import SupervisedRunner
from torch.utils.data import DataLoader
from catalyst.dl.callbacks import CriterionCallback
from catalyst.core.callbacks import EarlyStoppingCallback
from catalyst.core.callbacks import MetricAggregationCallback
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import timm
import numpy as np
from sklearn.metrics import cohen_kappa_score
from catalyst.core import Callback, CallbackOrder, State
from collections import defaultdict
from .models import ClassifcationMultiCropModel
from .models import ClassifcationMultiCropMultiHeadModel
from .losses import QWKLoss
from catalyst.utils import prepare_cudnn, set_global_seed
from catalyst.contrib.nn.optimizers import RAdam, Lookahead


def runTraining(params, *args, **kwargs):
    # Quite redundant, but it
    # will do the trick
    SEED = 42
    set_global_seed(SEED)
    prepare_cudnn(deterministic=True)
    dataset_train = ClassifcationDatasetMultiCrop(
        params['train_csv'],
        params['train_transformations'],
        params['train_image_dir'],
        params['train_mask_dir'],
        **params['dataset_config'])
    dataset_val = ClassifcationDatasetMultiCrop(
        params['val_csv'],
        params['val_transformations'],
        params['val_image_dir'],
        params['val_mask_dir'],
        **params['dataset_config'])

    train_loader = DataLoader(
        dataset_train,
        batch_size=params['batch_size'],
        num_workers=params['n_workers'],
        pin_memory=True,
        shuffle=True
    )
    validation_loader = DataLoader(
        dataset_val,
        batch_size=params['batch_size'],
        num_workers=params['n_workers'],
        pin_memory=True,
        shuffle=False
    )

    criterion = get_loss(
        params['loss_name'],
        **params['loss_config'])
    model = ClassifcationMultiCropModel(
        params['model_name'],
        **params['model_config'])
    if "load_weights" in params:
        print('loading weights')
        model.load_state_dict(torch.load(params['load_weights'])['model_state_dict'])
    model.cuda()
    optimizer = get_optimizer(
        params['optimizer_name'],
        model,
        **params['optimizer_config'])
    scheduler = get_scheduler(
        params['scheduler_name'],
        optimizer,
        **params['scheduler_config'])

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = validation_loader
    losses = dict({'loss_classification': criterion})
    runner = SupervisedRunner(
        input_key='features',
        input_target_key='targets'
        )
    callbacks = [
        CriterionCallback(
            input_key="targets",
            prefix="loss",
            criterion_key='loss_classification',
            multiplier=1.0
            ),
        QWKCallback(input_key="targets",
                    **params['qwk_config'])
        ]
    log_dir = params['log_dir']
    runner.train(
        model=model,
        criterion=losses,
        scheduler=scheduler,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders=loaders,
        logdir=log_dir,
        main_metric='loss',
        num_epochs=params['num_epochs'],
        verbose=True,
        minimize_metric=True
        )


def runTrainingMultiHead(params, *args, **kwargs):
    # Quite redundant, but it
    # will do the trick
    SEED = 42
    set_global_seed(SEED)
    prepare_cudnn(deterministic=True)
    dataset_train = ClassifcationDatasetMultiCropMultiHead(
        params['train_csv'],
        params['train_transformations'],
        params['train_image_dir'],
        params['train_mask_dir'],
        **params['dataset_config'])
    dataset_val = ClassifcationDatasetMultiCropMultiHead(
        params['val_csv'],
        params['val_transformations'],
        params['val_image_dir'],
        params['val_mask_dir'],
        **params['dataset_config'])

    train_loader = DataLoader(
        dataset_train,
        batch_size=params['batch_size'],
        num_workers=params['n_workers'],
        pin_memory=True,
        shuffle=True
    )
    validation_loader = DataLoader(
        dataset_val,
        batch_size=params['batch_size'],
        num_workers=params['n_workers'],
        pin_memory=True,
        shuffle=False
    )

    criterion = get_loss(
        params['loss_name'],
        **params['loss_config'])
    model = ClassifcationMultiCropMultiHeadModel(
        params['model_name'],
        **params['model_config'])
    if "load_weights" in params:
        print('loading weights')
        model.load_state_dict(torch.load(params['load_weights'])['model_state_dict'])
    model.cuda()
    optimizer = get_optimizer(
        params['optimizer_name'],
        model,
        **params['optimizer_config'])
    scheduler = get_scheduler(
        params['scheduler_name'],
        optimizer,
        **params['scheduler_config'])

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = validation_loader
    losses = dict({
        'loss_isup': criterion,
        'loss_gleason_major': criterion,
        'loss_gleason_minor': criterion
        })
    runner = SupervisedRunner(
        input_key='features',
        input_target_key=['targets_isup', 'targets_gleason_major', 'targets_gleason_minor'],
        output_key=["logits_isup", "logits_gleason_major", "logits_gleason_minor"]
        )
    callbacks = [
        CriterionCallback(
            input_key="targets_isup",
            prefix="loss_isup",
            output_key="logits_isup",
            criterion_key='loss_isup',
            multiplier=1.0
            ),
        CriterionCallback(
            input_key="targets_gleason_major",
            prefix="loss_gleason_major",
            output_key="logits_gleason_major",
            criterion_key='loss_gleason_major',
            multiplier=1.0
            ),
        CriterionCallback(
            input_key="targets_gleason_minor",
            prefix="loss_gleason_minor",
            output_key="logits_gleason_minor",
            criterion_key='loss_gleason_minor',
            multiplier=1.0
            ),
        QWKCallback(
            input_key="targets_isup",
            output_key="logits_isup",
            prefix='qwk_isup',
            **params['qwk_config']),
        QWKCallback(
            input_key="targets_gleason_major",
            output_key="logits_gleason_major",
            prefix='qwk_gleason_major',
            **params['qwk_config']),
        QWKCallback(
            input_key="targets_gleason_minor",
            output_key="logits_gleason_minor",
            prefix='qwk_gleason_minor',
            **params['qwk_config']),
        MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum",
            metrics={
                "loss_isup": 1.0,
                "loss_gleason_major": 0.75,
                "loss_gleason_minor": 0.75},
            ),
        EarlyStoppingCallback(patience=15)
        ]
    log_dir = params['log_dir']
    runner.train(
        model=model,
        criterion=losses,
        scheduler=scheduler,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders=loaders,
        logdir=log_dir,
        main_metric='loss',
        num_epochs=params['num_epochs'],
        verbose=True,
        minimize_metric=True
        )


def get_loss(loss_name, **kwarg):
    if loss_name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss(**kwarg)
    elif loss_name == 'MSE':
        return torch.nn.MSELoss(**kwarg)
    elif loss_name == 'SmoothL1Loss':
        return torch.nn.SmoothL1Loss(**kwarg)
    elif loss_name == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss(**kwarg)
    elif loss_name == 'QWKLoss':
        return QWKLoss(**kwarg)
    else:
        raise NotImplementedError


def get_model(model_name, **kwarg):
    model = timm.create_model(
        model_name,
        **kwarg)
    return model


def get_optimizer(optimizer_name, model, **kwarg):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **kwarg)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), **kwarg)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **kwarg)
    elif optimizer_name == 'RAdam':
        optimizer = RAdam(model.parameters(), **kwarg)
        optimizer = Lookahead(optimizer)
    else:
        raise NotImplementedError
    return optimizer


def get_scheduler(scheduler_name, optimizer, **kwarg):
    if scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, **kwarg)
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, **kwarg)
    elif scheduler_name == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer, **kwarg)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, **kwarg)
    else:
        raise NotImplementedError
    return scheduler


def quadratic_weighted_kappa(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = None,
    activation: str = None
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: quadratic kappa score
    """
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    outputs_clipped = list()
    outputs_clipped = np.rint(outputs)
    outputs_clipped[outputs_clipped < 0] = 0
    outputs_clipped[outputs_clipped > 5] = 5
    score = cohen_kappa_score(outputs_clipped, targets, weights='quadratic')
    return score


def quadratic_weighted_kappa_clf(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = None,
    activation: str = None
):
    outputs = torch.argmax(torch.nn.Softmax(dim=1)(outputs.detach()), dim=1)
    outputs = outputs.cpu().numpy()
    targets = targets.detach().cpu().numpy()
    score = cohen_kappa_score(outputs, targets  )
    return score


def quadratic_weighted_kappa_mt(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = None,
    activation: str = None
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: quadratic kappa score
    """
    outputs = outputs.detach().cpu().numpy()[:, 0]
    targets = targets.detach().cpu().numpy()
    outputs_clipped = list()
    outputs_clipped = np.rint(outputs)
    outputs_clipped[outputs_clipped < 0] = 0
    outputs_clipped[outputs_clipped > 5] = 5
    score = cohen_kappa_score(outputs_clipped, targets, weights='quadratic')
    return score


def quadratic_weighted_kappa_ord(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = None,
    activation: str = None
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: quadratic kappa score
    """
    activation_fn = torch.sigmoid
    outputs = activation_fn(outputs)
    outputs_sum = (outputs.cpu().detach().numpy() >= 0.5).sum(axis=1).astype(int)
    targets_sum = targets.detach().cpu().numpy().sum(axis=1).astype(int)
    score = cohen_kappa_score(
        outputs_sum,
        targets_sum,
        weights='quadratic')
    return score


def quadratic_weighted_kappa_ohe_class(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = None,
    activation: str = None
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: quadratic kappa score
    """
    activation_fn = torch.sigmoid
    outputs = activation_fn(outputs)
    targets = torch.argmax(targets, dim=1)
    outputs = torch.argmax(outputs, dim=1)
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    score = cohen_kappa_score(
        outputs,
        targets,
        weights='quadratic')
    return score


def get_qwk_mf_by_name(qwk_name):
    if qwk_name == 'ordinal':
        return quadratic_weighted_kappa_ord
    elif qwk_name == 'mt':
        return quadratic_weighted_kappa_mt
    elif qwk_name == 'clf':
        return quadratic_weighted_kappa_clf
    elif qwk_name == 'simple':
        return quadratic_weighted_kappa
    elif qwk_name == 'ohe_class':
        return quadratic_weighted_kappa_ohe_class
    else:
        raise NotImplementedError


class QWKCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        activation: str = "Sigmoid",
        qwk_name: str = 'simple',
            prefix: str = "qwk"):
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.predictions = defaultdict(lambda: [])
        self.metric_fn = get_qwk_mf_by_name(qwk_name)

    def on_epoch_start(self, state) -> None:
        self.accum = []

    def on_loader_start(self, state: State):
        self.predictions = defaultdict(lambda: [])

    def on_batch_end(self, state: State) -> None:
        targets = state.input[self.input_key]
        outputs = state.output[self.output_key]
        self.predictions[self.input_key].append(targets.detach().cpu())
        self.predictions[self.output_key].append(outputs.detach().cpu())
        metric = self.metric_fn(outputs, targets)
        state.batch_metrics[f"batch_{self.prefix}"] = metric

    def on_loader_end(self, state) -> None:
        self.predictions = {
            key: torch.cat(value, dim=0)
            for key, value in self.predictions.items()
        }
        targets = self.predictions[self.input_key]
        outputs = self.predictions[self.output_key]
        value = self.metric_fn(
            outputs, targets
        )
        state.loader_metrics[self.prefix] = value
