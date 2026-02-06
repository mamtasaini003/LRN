from .pde_datasets import (
    BurgersDataset,
    Burgers2dDataset,
    DarcyDataset,
    NavierStokesDataset,
    create_dataloaders
)
from .neuralop_loaders import (
    load_darcy,
    load_navier_stokes,
    load_burgers,
    create_neuralop_dataloaders,
    get_dataset_info,
    NEURALOP_AVAILABLE,
)

__all__ = [
    # Original datasets
    'BurgersDataset',
    'Burgers2dDataset',
    'DarcyDataset',
    'NavierStokesDataset',
    'create_dataloaders',
    # NeuralOperator loaders
    'load_darcy',
    'load_navier_stokes',
    'load_burgers',
    'create_neuralop_dataloaders',
    'get_dataset_info',
    'NEURALOP_AVAILABLE',
]

