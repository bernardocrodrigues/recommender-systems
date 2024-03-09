""" binaps_wrapper.py

This module contains a series of helper functions to run the BinaPs algorithm through its CLI. It
also contains a function to parse the output of the BinaPs algorithm. This way, we easily integrate
BinaPs original implementation with our own modules.

Copyright 2022 Bernardo C. Rodrigues
See LICENSE file for license details


"""

import re
import os
from pathlib import Path
from subprocess import run
from typing import List, Optional, TextIO, Tuple
import torch
from tabulate import tabulate
from torch.nn import init
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from pydantic import Field, validate_call, ConfigDict
from typing_extensions import Annotated
import numpy as np

from pattern_mining.binaps.original.Binaps_code import dataLoader as mydl
from pattern_mining.binaps.original.Binaps_code import my_loss as mylo
from pattern_mining.binaps.original.Binaps_code.network import initWeights, Net
from pattern_mining.binaps.original.Binaps_code.my_layers import BinarizeTensorThresh


def generate_synthetic_data(
    output_path: Path,
    row_quantity,
    column_quantity,
    file_prefix: str,
    max_pattern_size: int,
    noise: float = 0.5,
    density: float = 0.5,
) -> None:
    """Generates a synthetic database based on known patterns.

    This is a wrapper function over the vanilla R script generate_toy.R. It places 4 files at the
    current work directory:
        <file_prefix>.dat - row_quantity x column_quantity dataset
        <file_prefix>.dat_patterns.txt - real patterns in <file_prefix>.dat
        <file_prefix>_itemOverlap.dat row_quantity x column_quantity dataset
        <file_prefix>_itemOverlap.dat_patterns.txt - real patterns in <file_prefix>_itemOverlap.dat

    Arbitrary noise is added to test the BinaPS robustness. In <file_prefix>_itemOverlap.dat the
    patterns may overlap (e.g ABC CDE CEF) while in <file_prefix>.dat they may not (e.g. AB C DE F).

    Args:
        row_quantity: number of rows, or transactions, that the synthetic dataset should have
        column_quantity: number of columns that the synthetic dataset should have. In other words,
            how many possible atributes, or items, each row may have.
        file_prefix: prefix that should be present on all generated files.
        max_pattern_size: biggest pattern size allowed to be used generated
        noise: percentage of the dataset to be flipped randomly
        density: percentage of the dataset that is non-zero
    """

    assert max_pattern_size > 0
    assert 0 <= noise <= 1
    assert 0 <= density <= 1

    source_root_dir = os.environ.get("SOURCE_ROOT_DIR")

    print(f"Generating synthetic data at {output_path}")
    os.chdir(output_path)

    cmd = (
        f"Rscript {source_root_dir}/pattern_mining/binaps/original/Data/Synthetic_data/generate_toy.R AND "
        f"{column_quantity} {row_quantity} {max_pattern_size} {file_prefix} {noise} {density}"
    )

    print(cmd)
    output = run(cmd.split(" "), capture_output=True, check=True)
    print(output.stdout.decode())


def run_binaps_cli(
    data_path: str,
    train_set_size: float = 0.9,
    batch_size: int = 64,
    test_batch_size: int = 64,
    epochs: int = 10,
    learning_rate: float = 0.01,
    weight_decay: float = 0,
    gamma: float = 0.1,
    seed: int = 1,
    hidden_dimension: int = -1,
) -> None:
    """Runs BinaPs Autoencoder

    This will ingest a given dataset and extract its patterns through BinaPs. This will place a
    .binaps.patterns file on the current working dir with the pattern list

    Args:
        data_path: path for a .dat format database. generate_synthetic_data provides data in the
            expected format, use it as template.
        hidden_dimension: number of neurons in the hidden dimension. This also indicates the number
            of inferred patterns. Each neuron will correspond to a single pattern.
        epochs: how many epochs will the neural network train
    """

    source_root_dir = os.environ.get("SOURCE_ROOT_DIR")

    cmd = (
        f"python3 {source_root_dir}pattern_mining/binaps/original/Binaps_code/main.py -i={data_path} "
        f"--train_set_size={train_set_size} "
        f"--batch_size={batch_size} "
        f"--test_batch_size={test_batch_size} "
        f"--epochs={epochs} "
        f"--lr={learning_rate} "
        f"--weight_decay={weight_decay} "
        f"--gamma={gamma} "
        f"--seed={seed} "
        f"--hidden_dim={hidden_dimension}"
    )

    print(cmd)
    output = run(cmd.split(" "), capture_output=True, check=True)
    stdout = output.stdout.decode()

    print_character_length = 300
    print(f"{stdout[:print_character_length]} [output truncated]")
    print("...")
    print(f"[output truncated] {stdout[-print_character_length:]}")


@validate_call(config=ConfigDict(strict=True))
def run_binaps(
    input_dataset_path: Annotated[str, Field(min_length=1)],
    train_set_size: Annotated[float, Field(ge=0, le=1)] = 0.9,
    batch_size: Annotated[int, Field(gt=0)] = 64,
    test_batch_size: Annotated[int, Field(gt=0)] = 64,
    epochs: Annotated[int, Field(gt=0)] = 10,
    learning_rate: Annotated[float, Field(gt=0)] = 0.01,
    weight_decay: Annotated[float, Field(ge=0)] = 0,
    gamma: Annotated[float, Field(ge=0)] = 0.1,
    seed: int = 1,
    hidden_dimension: Optional[Annotated[int, Field(gt=0)]] = None,
):
    """
    Run the Binaps algorithm on a given dataset.

    Args:
        input_dataset_path (str): The path to the input dataset.
        train_set_size (float): The size of the training set as a fraction of the dataset
                                (default: 0.9).
        batch_size (int): The batch size for training (default: 64).
        test_batch_size (int): The batch size for testing (default: 64).
        epochs (int): The number of training epochs (default: 10).
        learning_rate (float): The learning rate for optimization (default: 0.01).
        weight_decay (float): Weight decay for optimization (default: 0).
        gamma (float): Gamma value for optimization (default: 0.1).
        seed (int): Random seed for reproducibility (default: 1).
        hidden_dimension (int): Hidden dimension for the NN. If None, the number of neurons in the
                                hidden layer will be equal to the number of columns in the dataset
                                (default: None).

    Returns:
        Tuple[torch.Tensor, List[float], List[float]]: A tuple containing weights,
                                                       training losses, and test losses.

    Example:
        weights, training_losses, test_losses = run_binaps(
            input_dataset_path="my_dataset.dat",
            train_set_size=0.8,
            batch_size=32,
            epochs=15,
            learning_rate=0.001,
            weight_decay=0.001,
        )
    """
    torch.manual_seed(seed)
    torch.set_num_threads(16)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hidden_dimension_arg = -1 if hidden_dimension is None else hidden_dimension

    weights, training_losses, test_losses = learn(
        input_dataset_path,
        learning_rate,
        gamma,
        weight_decay,
        epochs,
        hidden_dimension_arg,
        train_set_size,
        batch_size,
        test_batch_size,
        device,
        device,
    )

    return weights, training_losses, test_losses


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def get_patterns_from_weights(
    weights: torch.Tensor, threshold: Annotated[float, Field(ge=0)]
) -> List[np.ndarray]:
    """
    Extract patterns from a set of BinaPs weights using a given threshold.

    Args:
        weights (torch.Tensor): The BinaPs weights from which patterns will be extracted.
        threshold (float): The threshold for pattern extraction (default: 0.2).

    Returns:
        List[np.ndarray]: A list of patterns represented as NumPy arrays.

    Example:
        weights = torch.tensor([0.1, 0.3, 0.6, 0.2, 0.4, 0.8])
        threshold = 0.4
        patterns = get_patterns_from_weights(weights, threshold)
        # Output: [array([2, 5]), array([4, 5])]

    """
    patterns = []

    with torch.no_grad():
        for hn in BinarizeTensorThresh(weights, threshold):  # pylint: disable=invalid-name
            pattern = torch.squeeze(hn.nonzero())
            if hn.sum() >= 2:
                patterns.append(pattern.cpu().numpy())

    return patterns


def parse_binaps_patterns(file_object: TextIO) -> List[List[int]]:
    """
    Parse binaps patterns from a file.

    Args:
        file_object: A file-like object containing the binaps detected patterns.

    Returns:
        A list of lists representing the parsed binaps patterns.

    This function reads the contents of the provided file and extracts binaps patterns.
    Each binaps pattern is represented as a list of integers. The function returns a list
    of these patterns.

    Example:
        patterns_file = open('patterns.txt', 'r')
        patterns = parse_binaps_patterns(patterns_file)
        patterns_file.close()
    """

    file_contents = file_object.read()

    regex = re.compile(r"\[[ ]*([ \d\n]+)]")
    patterns_as_strings = regex.findall(file_contents)

    patterns = []
    for pattern_as_string in patterns_as_strings:
        pattern = [int(d) for d in pattern_as_string.replace("\n", " ").split()]
        patterns.append(pattern)

    return patterns


def compare_datasets_based_on_f1(estimated_patterns_file: str, real_patterns_file: str):
    """Get the F1 Score on the inferred dataset.

    Args:
        estimated_patterns_file: path for a BinaPs' .binaps.patterns output file
        real_patterns_file: path for a .dat_patterns.txt patterns file as generated by
            generate_synthetic_data.
    """

    source_root_dir = os.environ.get("SOURCE_ROOT_DIR")

    cmd = (
        f"python3 {source_root_dir}binaps/original/Data/Synthetic_data/comp_patterns.py "
        f"-p {estimated_patterns_file} "
        f"-t Binaps -r {real_patterns_file} -m F1"
    )

    output = run(cmd.split(" "), capture_output=True, check=True)
    return float(output.stdout.decode())


def display_as_table(
    data: List[List], headers: Optional[List[str]] = None, title: Optional[str] = None
) -> None:
    """
    Display data as a formatted table.

    Args:
        data: A list of lists representing the data to be displayed.
        headers (optional): A list of strings representing the column headers (default: []).
        title (optional): A string representing the title of the table (default: None).

    Returns:
        None

    This function takes the provided data and displays it as a formatted table. The data should be
    provided as a list of lists, where each inner list represents a row of the table. The column
    headers, if provided, should be specified as a list of strings. If a title is provided, it will
    be displayed above the table.

    Example:
        data = [
            ['John Doe', 25, 'Engineer'],
            ['Jane Smith', 32, 'Manager'],
            ['Mark Johnson', 41, 'Developer']
        ]

        headers = ['Name', 'Age', 'Role']

        display_as_table(data, headers=headers, title='Employee Information')

        # Output:
        # Employee Information
        # ╒═════════════╤═════╤═════════════╕
        # │ Name        │ Age │ Role        │
        # ╞═════════════╪═════╪═════════════╡
        # │ John Doe    │ 25  │ Engineer    │
        # ├─────────────┼─────┼─────────────┤
        # │ Jane Smith  │ 32  │ Manager     │
        # ├─────────────┼─────┼─────────────┤
        # │ Mark Johnson│ 41  │ Developer   │
        # ╘═════════════╧═════╧═════════════╛
    """
    if title:
        print(title)
    print(tabulate(data, headers=headers, tablefmt="fancy_grid"))


def learn(
    input_file: str,
    learn_rate: float,
    gamma: float,
    weight_decay: float,
    epochs: int,
    hidden_dim: int,
    train_set_size: float,
    batch_size: int,
    test_batch_size: int,
    device_cpu: torch.device,
    device_gpu: torch.device,
):
    """
    Train the Binaps algorithm on a given dataset.

    This function is based on the learn function from the original BinaPs implementation found at
    binaps/original/Binaps_code/network.py:learn

    Args:
        input_file (str): The path to the input dataset. The dataset must be in the .dat format.
        learn_rate (float): The learning rate for optimization.
        gamma (float): Gamma value for optimization.
        weight_decay (float): Weight decay for optimization.
        epochs (int): The number of training epochs.
        hidden_dim (int): Hidden dimension for the Binaps algorithm. If -1, the number of neurons
                            in the hidden layer will be equal to the number of columns in the
                            dataset.
        train_set_size (float): The size of the training set as a fraction of the dataset. Must
                                be between 0 and 1.
        batch_size (int): The batch size for training.
        test_batch_size (int): The batch size for testing. Must be between 0 and 1.
        device_cpu (torch.device): The CPU device to be used.
        device_gpu (torch.device): The GPU device to be used.

    Returns:
        Tuple[torch.Tensor, List[float], List[float]]: A tuple containing weights,
                                                       training losses, and test losses.

    Example:
        weights, training_losses, test_losses = learn(
            input_file="my_dataset.dat",
            learn_rate=0.01,
            gamma=0.1,
            weight_decay=0,
            epochs=15,
            hidden_dim=100,
            train_set_size=0.8,
            batch_size=32,
            test_batch_size=32,
            device_cpu=torch.device("cpu"),
            device_gpu=torch.device("cuda"),
            )
    """

    kwargs = {}
    train_dataset = mydl.DatDataset(input_file, train_set_size, True, device_cpu)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        mydl.DatDataset(input_file, train_set_size, False, device_cpu),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )

    if hidden_dim == -1:
        hidden_dim = train_dataset.ncol()

    weights = torch.zeros(hidden_dim, train_dataset.ncol(), device=device_gpu)
    initWeights(weights, train_dataset.data)
    weights.clamp_(1 / (train_dataset.ncol()), 1)
    bias_init = torch.zeros(hidden_dim, device=device_gpu)
    init.constant_(bias_init, -1)

    model = Net(weights, bias_init, train_dataset.getSparsity(), device_cpu, device_gpu).to(
        device_gpu
    )
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    loss_function = mylo.weightedXor(train_dataset.getSparsity(), weight_decay, device_gpu)

    scheduler = MultiStepLR(optimizer, [5, 7], gamma=gamma)

    training_losses = []
    test_losses = []

    for _ in range(epochs):
        training_loss = train(model, device_gpu, train_loader, optimizer, loss_function)
        test_loss = test(model, device_gpu, test_loader, loss_function)

        scheduler.step()
        test_loss_numpy = test_loss.cpu().numpy()

        training_losses.append(training_loss)
        test_losses.append(test_loss_numpy)

    return weights, training_losses, test_losses


def test(
    model: Net,
    torch_device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    loss_function: callable,
) -> Tuple[torch.Tensor, int]:
    """
    Test the model on a given dataset.

    This function is based on the test function from the original BinaPs implementation found at
    binaps/original/Binaps_code/network.py:test

    Args:
        model (Net): The model to be tested.
        torch_device (torch.device): The GPU device to be used.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        loss_function (callable): The loss function to be used.

    Returns:
        int: The test loss.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(torch_device)
            output = model(data)
            iteration_ew = [
                par for name, par in model.named_parameters() if name.endswith("enc.weight")
            ]
            test_loss += loss_function(output, data, next(iter(iteration_ew)))

    return test_loss


def train(model, torch_device, train_loader, optimizer, loss_function):
    """
    Train the model on a given dataset.

    This function is based on the train function from the original BinaPs implementation found at
    binaps/original/Binaps_code/network.py:train

    Args:
        model (Net): The model to be trained.
        torch_device (torch.device): The GPU device to be used.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        loss_function (callable): The loss function to be used.

    Returns:
        int: The training loss.
    """
    model.train()
    for _, (data, _) in enumerate(train_loader):
        data = data.to(torch_device)
        optimizer.zero_grad()
        output = model(data)
        iteration_ew = [
            par for name, par in model.named_parameters() if name.endswith("enc.weight")
        ]
        loss = loss_function(output, data, next(iter(iteration_ew)))
        loss.backward()
        optimizer.step()
        model.clipWeights()

    return loss.item()
