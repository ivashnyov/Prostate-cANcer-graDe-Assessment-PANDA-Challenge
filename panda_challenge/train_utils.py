from .dataset import ClassifcationDatasetSimpleTrain
import collections
from catalyst.dl.runner import SupervisedRunner
from torch.utils.data import DataLoader
from catalyst.dl.callbacks import CriterionCallback
from catalyst.dl.callbacks import MeterMetricsCallback
from catalyst.utils import meters


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

    criterion = get_loss(params['loss_name'], *args, **kwargs)
    model = get_model(params['model_name'], *args, **kwargs)
    optimizer = get_optimizer(params['optimizer_name'], *args, **kwargs)
    scheduler = get_scheduler(params['scheduler_name'], *args, **kwargs)

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = validation_loader
    losses = dict({'loss_classification': criterion})
    runner = SupervisedRunner(
        input_key='features',
        output_key=None,
        input_target_key='targets'
        )
    callbacks = [
        CriterionCallback(
            input_key="targets",
            output_key='logits',
            prefix="loss",
            criterion_key='loss_classification',
            multiplier=1.0
            )
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


def get_loss():
    pass


def get_model():
    pass


def get_optimizer():
    pass


def get_scheduler():
    pass


def QWKCallback(MeterMetricsCallback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "qwk",
        num_classes: int = 6,
        activation: str = "Softmax",
    ):
        pass
