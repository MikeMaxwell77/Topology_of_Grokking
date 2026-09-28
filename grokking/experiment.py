"""Orchestration for the staged grokking experiment."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .config import ExperimentConfig
from .data import ModularArithmeticDataset
from .network import TinyTransformer
from .topology import analyze_topology_all_layers, compute_dataset_topology
from .training import evaluate, train_epoch
from .agreement import load_agreement
from .entropy import spectral_entropy


def build_datasets(config: ExperimentConfig, exponent: int):
    common = dict(p=config.modulus, c=exponent, r=config.input_range,
                  train_fraction=config.train_fraction, seed=config.seed)
    return (ModularArithmeticDataset(train=True, **common),
            ModularArithmeticDataset(train=False, **common))


def build_loaders(train_data, validation_data, config: ExperimentConfig):
    generator = torch.Generator().manual_seed(config.seed)
    return (
        DataLoader(train_data, batch_size=config.batch_size, shuffle=True, generator=generator),
        DataLoader(validation_data, batch_size=config.batch_size, shuffle=False),
    )


def parameter_norms(model: torch.nn.Module) -> tuple[float, float]:
    flattened = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    return flattened.norm(1).item(), flattened.norm(2).item()


def save_history(history: dict, path: str | Path = "grokking_history.pkl") -> None:
    with Path(path).open("wb") as stream:
        pickle.dump(history, stream)


def run_experiment(
    config: ExperimentConfig = ExperimentConfig(),
    *,
    device: torch.device | str | None = None,
    stage_exponents: tuple[int, ...] | None = None,
    stage_threshold: float = 0.95,
    topology_fn=analyze_topology_all_layers,
    dataset_topology_fn=compute_dataset_topology,
):
    """Run training while allowing expensive analysis functions to be mocked."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.seed)
    if config.task == "simple_agreement" and stage_exponents is not None:
        raise ValueError("arithmetic stages do not apply to Simple Agreement")
    exponents = (
        stage_exponents
        if stage_exponents is not None
        else (config.exponent, *(value for value in (2, 3) if value > config.exponent))
    )
    if not exponents:
        raise ValueError("stage_exponents cannot be empty")
    extra_loaders = {}
    dataset_metadata = None
    if config.task == "simple_agreement":
        exponents = (config.exponent,)
        datasets, dataset_metadata = load_agreement(config.agreement_data_dir)
        train_data, validation_data = datasets["train"], datasets["val"]
        extra_loaders = {key: DataLoader(datasets[key], batch_size=config.batch_size)
                         for key in ("g1_test", "g2_test")}
        model_options = dict(vocab_size=len(dataset_metadata["vocabulary"]),
                             sequence_length=6, num_classes=2, padding_idx=0)
    else:
        train_data, validation_data = build_datasets(config, exponents[0])
        model_options = dict(vocab_size=max(config.input_range + 2, config.modulus + 1, max(exponents) + 1))
    train_loader, validation_loader = build_loaders(train_data, validation_data, config)
    full_data = ConcatDataset([train_data, validation_data])
    ideal_topology = (dataset_topology_fn(full_data, config.modulus, exponents[0], max_samples=2_000)
                      if config.task == "modular" and config.tda_interval else None)

    model = TinyTransformer(**model_options, d_model=config.d_model,
                            n_heads=config.n_heads, n_layers=config.n_layers, d_ff=config.d_ff).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    history = {
        "config": asdict(config),
        "model_options": model_options,
        "dataset": dataset_metadata,
        "optimizer_step": [],
        "parameter_l1": [],
        "parameter_l2": [],
        "generalization_acc": [],
        "linear_rule_acc": [],
        "entropy": [],
        "epoch": [],
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "stage": [],
        "exponent": [],
        "tda_computed": [],
        "topology": [],
        "ideal_topology": ideal_topology,
    }
    previous_topology = None
    stage = 0
    optimizer_step = 0
    probe_indices = np.random.default_rng(config.seed).choice(
        len(train_data), min(len(train_data), config.entropy_probe_size), replace=False).tolist()
    probe_loader = DataLoader(Subset(train_data, probe_indices), batch_size=config.batch_size)
    history["entropy_definition"] = {
        "spectral": "centered covariance X.T X / N; float64; natural log; normalized by log(d_model)",
        "representation": "final sequence position after each transformer layer, before output head",
        "probe": "fixed initial-stage training subset", "probe_indices": probe_indices,
        "persistent": "positive finite persistence lifetimes; natural log; normalized by log(positive bar count)",
        "zero_mass": "undefined (NaN)",
        "topology_preprocessing": "fixed validation subsample, coordinate standardization; see compute_topology",
    }

    def record_entropy(epoch):
        was_training = model.training
        model.eval()
        try:
            layers = {}
            with torch.no_grad():
                for layer in range(config.n_layers):
                    states = np.concatenate([model.get_hidden_states(inputs.to(device), layer).cpu().numpy()
                                             for inputs, _ in probe_loader])
                    layers[layer] = spectral_entropy(states)
            l1, l2 = parameter_norms(model)
            history["entropy"].append(dict(epoch=epoch, optimizer_step=optimizer_step,
                                           layers=layers, parameter_l1=l1, parameter_l2=l2))
        finally:
            model.train(was_training)

    record_entropy(-1)

    for epoch in range(config.num_epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device)
        optimizer_step += len(train_loader)
        if epoch % config.entropy_interval == 0 or epoch == config.num_epochs - 1:
            record_entropy(epoch)
        should_log = epoch % config.log_interval == 0
        should_compute_topology = bool(config.tda_interval) and epoch % config.tda_interval == 0
        should_log = should_log or epoch == config.num_epochs - 1
        if not (should_log or should_compute_topology):
            continue
        validation_accuracy = evaluate(model, validation_loader, device)
        extra_accuracy = {key: evaluate(model, loader, device) for key, loader in extra_loaders.items()}
        l1_norm, l2_norm = parameter_norms(model)
        if should_log:
            print(f"Epoch {epoch:5d} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_accuracy:.4f} | Val Acc: {validation_accuracy:.4f} | "
                  f"L1: {l1_norm:.4f} | L2: {l2_norm:.4f}")
            if extra_accuracy:
                print(f"  Grammatical generalization: {extra_accuracy['g1_test']:.4f} | "
                      f"Linear-rule diagnostic: {extra_accuracy['g2_test']:.4f}")
        history["optimizer_step"].append(optimizer_step)
        history["parameter_l1"].append(l1_norm)
        history["parameter_l2"].append(l2_norm)
        history["generalization_acc"].append(extra_accuracy.get("g1_test"))
        history["linear_rule_acc"].append(extra_accuracy.get("g2_test"))
        history["epoch"].append(epoch)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(validation_accuracy)
        history["train_loss"].append(train_loss)
        history["stage"].append(stage)
        history["exponent"].append(exponents[stage] if config.task == "modular" else None)
        history["tda_computed"].append(should_compute_topology)

        if should_compute_topology:
            current_topology = topology_fn(model, validation_loader, device, config.n_layers,
                                           prev_topology=previous_topology,
                                           ideal_topology=ideal_topology, maxdim=2,
                                           max_samples=config.tda_max_samples)
            previous_topology = current_topology
        elif history["topology"]:
            current_topology = history["topology"][-1]
        else:
            current_topology = {layer: {} for layer in range(config.n_layers)}
        history["topology"].append(current_topology)

        if validation_accuracy > stage_threshold and stage + 1 < len(exponents):
            stage += 1
            new_train, new_validation = build_datasets(config, exponents[stage])
            train_data = ConcatDataset([train_data, new_train])
            validation_data = ConcatDataset([validation_data, new_validation])
            train_loader, validation_loader = build_loaders(train_data, validation_data, config)
            previous_topology = None
            full_data = ConcatDataset([train_data, validation_data])
            ideal_topology = (dataset_topology_fn(full_data, config.modulus, exponents[stage],
                                                 max_samples=2_000) if config.tda_interval else None)
            history["ideal_topology"] = ideal_topology

    return model, history


def _comma_separated_integers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one exponent is required")
    return result


def parse_args(argv=None) -> argparse.Namespace:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Train a transformer and track activation topology.")
    parser.add_argument("--task", choices=("modular", "simple_agreement"), default=defaults.task)
    parser.add_argument("--agreement-data-dir", default=defaults.agreement_data_dir)
    parser.add_argument("--entropy-interval", type=int, default=defaults.entropy_interval,
                        help="spectral entropy cadence in epochs (also records initialization and final epoch)")
    parser.add_argument("--entropy-probe-size", type=int, default=defaults.entropy_probe_size)
    parser.add_argument("--modulus", type=int, default=defaults.modulus)
    parser.add_argument("--exponent", type=int, default=defaults.exponent)
    parser.add_argument("--input-range", type=int, default=defaults.input_range)
    parser.add_argument("--d-model", type=int, default=defaults.d_model)
    parser.add_argument("--n-heads", type=int, default=defaults.n_heads)
    parser.add_argument("--n-layers", type=int, default=defaults.n_layers)
    parser.add_argument("--d-ff", type=int, default=defaults.d_ff)
    parser.add_argument("--train-fraction", type=float, default=defaults.train_fraction)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--log-interval", type=int, default=defaults.log_interval)
    parser.add_argument("--tda-interval", type=int, default=defaults.tda_interval)
    parser.add_argument("--tda-max-samples", type=int, default=defaults.tda_max_samples)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--stage-exponents", type=_comma_separated_integers)
    parser.add_argument("--stage-threshold", type=float, default=0.95)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def config_from_args(arguments: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        task=arguments.task,
        agreement_data_dir=arguments.agreement_data_dir,
        entropy_interval=arguments.entropy_interval,
        entropy_probe_size=arguments.entropy_probe_size,
        modulus=arguments.modulus,
        exponent=arguments.exponent,
        input_range=arguments.input_range,
        d_model=arguments.d_model,
        n_heads=arguments.n_heads,
        n_layers=arguments.n_layers,
        d_ff=arguments.d_ff,
        train_fraction=arguments.train_fraction,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        num_epochs=arguments.epochs,
        log_interval=arguments.log_interval,
        tda_interval=arguments.tda_interval,
        tda_max_samples=arguments.tda_max_samples,
        seed=arguments.seed,
    )


def main(argv=None) -> None:
    arguments = parse_args(argv)
    config = config_from_args(arguments)
    requested_device = None if arguments.device == "auto" else arguments.device
    model, history = run_experiment(
        config,
        device=requested_device,
        stage_exponents=arguments.stage_exponents,
        stage_threshold=arguments.stage_threshold,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = arguments.output_dir / "grokking_history.pkl"
    model_path = arguments.output_dir / "grokking_model.pt"
    save_history(history, history_path)
    torch.save(model.state_dict(), model_path)
    if not arguments.no_plots:
        from .plotting import save_tracking_plots

        save_tracking_plots(history, arguments.output_dir / "plots")
    print(f"Training complete. Outputs saved in {arguments.output_dir}.")


if __name__ == "__main__":
    main()
