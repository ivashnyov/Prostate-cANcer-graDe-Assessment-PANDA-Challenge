from .dataset import ClassifcationDatasetSimpleTrain
import collections
from catalyst.dl.runner import SupervisedRunner
from torch.utils.data import DataLoader
from catalyst.dl.callbacks import CriterionCallback
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import timm
import numpy as np
from sklearn.metrics import cohen_kappa_score
from catalyst.core import Callback, CallbackOrder, State
from collections import defaultdict
from catalyst.utils import get_activation_fn


def runTraining(params, *args, **kwargs):
    dataset_train = ClassifcationDatasetSimpleTrain(
        params['train_csv'],
        params['train_transformations'],
        params['train_image_dir'],
        params['train_mask_dir'])
    dataset_val = ClassifcationDatasetSimpleTrain(
        params['val_csv'],
        params['val_transformations'],
        params['val_image_dir'],
        params['val_mask_dir'])

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
    model = get_model(
        params['model_name'],
        **params['model_config'])
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


def get_loss(loss_name, **kwarg):
    if loss_name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss(**kwarg)
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
    score = cohen_kappa_score(outputs, targets, weights='quadratic')
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
    outputs_sum = (outputs.cpu().detach().numpy() >= 0.5).sum(axis=1)
    score = cohen_kappa_score(
        outputs_sum,
        targets.detach().cpu().numpy(),
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
            targets, outputs
        )      
        state.loader_metrics[self.prefix] = value
