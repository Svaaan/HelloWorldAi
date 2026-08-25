"""Tests for the real training path.

Requires torch. Skips cleanly when it is absent, so the other suites still run
on a machine without it. Multiple CPU "devices" stand in for multiple GPUs,
which exercises the sharding and gradient-reduction paths without needing one.

The headline test is test_uneven_shards_train_identically_to_one_big_batch:
proportional splitting is only safe if it is mathematically equivalent to
training on the whole batch, and that is what makes the pooling claim true.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    from backend.service.trainer import (
        PooledTrainer,
        build_workload,
        resolve_devices,
        train,
    )

MLP_SPEC = {"architecture": "mlp", "input_dim": 32, "hidden_dim": 64,
            "depth": 2, "output_dim": 5}
# dropout=0 so the run is deterministic; the shard-equivalence property is
# exact only for a deterministic model.
LM_SPEC = {"architecture": "transformer", "vocab_size": 64, "d_model": 32,
           "n_head": 2, "n_layer": 1, "seq_len": 8, "dropout": 0.0}


def _train_weights(shard_sizes, spec=MLP_SPEC, steps=5, seed=0, lr=0.1):
    """Train with a given shard layout and return the resulting weights."""
    workload = build_workload(spec)
    devices = [torch.device("cpu") for _ in shard_sizes]

    torch.manual_seed(seed)
    pooled = PooledTrainer(workload["factory"], devices)
    optimizer = torch.optim.SGD(pooled.parameters(), lr=lr)

    generator = torch.Generator().manual_seed(1234)
    for _ in range(steps):
        x, y = workload["make_batch"](sum(shard_sizes), generator, torch.device("cpu"))
        pooled.train_step(x, y, shard_sizes, workload["loss_fn"], optimizer)

    return [p.detach().clone() for p in pooled.parameters()]


# --- the correctness property the whole design rests on ------------------

def test_uneven_shards_train_identically_to_one_big_batch():
    """A 30/20/10 split must land on the same weights as a single batch of 60.

    Gradients are averaged weighted by sample count, so the split is exactly
    the gradient of the mean loss over the full batch. Only floating point
    reassociation should differ.
    """
    single = _train_weights([60])
    sharded = _train_weights([30, 20, 10])

    assert len(single) == len(sharded)
    for a, b in zip(single, sharded):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), \
            "max deviation %g" % (a - b).abs().max().item()


def test_a_lopsided_split_is_also_equivalent():
    # The realistic case: one fast card takes most of the batch.
    single = _train_weights([64])
    sharded = _train_weights([50, 10, 4])
    for a, b in zip(single, sharded):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_an_even_split_is_equivalent_too():
    single = _train_weights([60])
    sharded = _train_weights([20, 20, 20])
    for a, b in zip(single, sharded):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_the_primary_device_gradient_is_not_double_counted():
    """replicas[0] IS master, so an in-place reduction would count it twice.

    A two-way split would then diverge from the single-batch result.
    """
    single = _train_weights([40])
    sharded = _train_weights([20, 20])
    for a, b in zip(single, sharded):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


# --- training actually works --------------------------------------------

def test_loss_decreases_on_a_learnable_problem():
    torch.manual_seed(0)
    workload = build_workload(MLP_SPEC)
    pooled = PooledTrainer(workload["factory"], [torch.device("cpu")])
    optimizer = torch.optim.Adam(pooled.parameters(), lr=1e-2)

    generator = torch.Generator().manual_seed(7)
    x, y = workload["make_batch"](64, generator, torch.device("cpu"))

    first = pooled.train_step(x, y, [64], workload["loss_fn"], optimizer)
    for _ in range(40):
        last = pooled.train_step(x, y, [64], workload["loss_fn"], optimizer)

    assert last < first, "loss did not fall: %s -> %s" % (first, last)


def test_empty_batch_is_rejected():
    workload = build_workload(MLP_SPEC)
    pooled = PooledTrainer(workload["factory"], [torch.device("cpu")])
    optimizer = torch.optim.SGD(pooled.parameters(), lr=0.1)
    generator = torch.Generator().manual_seed(0)
    x, y = workload["make_batch"](4, generator, torch.device("cpu"))

    try:
        pooled.train_step(x, y, [0], workload["loss_fn"], optimizer)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_a_device_given_zero_samples_is_skipped_cleanly():
    single = _train_weights([30])
    with_idle = _train_weights([30, 0])
    for a, b in zip(single, with_idle):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


# --- workloads -----------------------------------------------------------

def test_mlp_workload_reports_parameters_and_cost():
    workload = build_workload(MLP_SPEC)
    assert workload["param_count"] > 0
    assert workload["flops_per_sample"] > 0


def test_transformer_workload_builds_and_trains():
    workload = build_workload(LM_SPEC)
    assert workload["param_count"] > 0

    torch.manual_seed(0)
    pooled = PooledTrainer(workload["factory"], [torch.device("cpu")])
    optimizer = torch.optim.Adam(pooled.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(3)
    x, y = workload["make_batch"](8, generator, torch.device("cpu"))

    loss = pooled.train_step(x, y, [8], workload["loss_fn"], optimizer)
    assert loss > 0 and loss == loss  # finite, not NaN


def test_dropout_is_configurable_for_deterministic_runs():
    import torch as _t
    model = build_workload(dict(LM_SPEC, dropout=0.0))["factory"]()
    rates = [m.p for m in model.modules() if isinstance(m, _t.nn.Dropout)]
    assert rates and all(r == 0.0 for r in rates), rates


def test_transformer_sharding_is_also_equivalent():
    single = _train_weights([16], spec=LM_SPEC, steps=3, lr=0.05)
    sharded = _train_weights([10, 6], spec=LM_SPEC, steps=3, lr=0.05)
    for a, b in zip(single, sharded):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-3)


def test_unknown_architecture_is_rejected():
    try:
        build_workload({"architecture": "quantum-hypercube"})
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- real data -----------------------------------------------------------

def _separable_dataset(rows=400, features=8, classes=3, seed=0):
    """A genuinely learnable dataset: each class sits around its own centroid."""
    import numpy as np
    rng = np.random.default_rng(seed)
    centroids = rng.normal(scale=4.0, size=(classes, features))
    labels = rng.integers(0, classes, size=rows)
    x = centroids[labels] + rng.normal(scale=0.5, size=(rows, features))
    return x.astype("float32"), labels.astype("int64")


def test_training_on_a_real_dataset_learns_it():
    """Loss must fall much further on learnable data than on noise.

    This is what proves the submitted dataset actually reaches the model
    rather than the trainer quietly using synthetic batches.
    """
    x, y = _separable_dataset()
    spec = {"architecture": "mlp", "hidden_dim": 64, "depth": 2}

    torch.manual_seed(0)
    real = train({"model_spec": spec,
                  "hyperparameters": {"steps": 60, "batch_size": 64, "learning_rate": 1e-2}},
                 lambda _m: None, plan=None, dataset=(x, y))["metrics"]

    assert real["dataset_rows"] == 400
    assert real["synthetic_data"] is False
    # Three balanced classes start near ln(3) = 1.10; separable data should go well below.
    assert real["final_loss"] < 0.5, real
    assert real["final_loss"] < real["initial_loss"] * 0.6, real


def test_model_dimensions_are_inferred_from_the_dataset():
    x, y = _separable_dataset(rows=120, features=11, classes=4)
    result = train({"model_spec": {"architecture": "mlp", "hidden_dim": 16, "depth": 1},
                    "hyperparameters": {"steps": 2, "batch_size": 16}},
                   lambda _m: None, plan=None, dataset=(x, y))

    first = result["state_dict"]["0.weight"]
    assert first.shape[1] == 11, first.shape        # input width from the data
    last = result["state_dict"]["2.weight"]
    assert last.shape[0] == 4, last.shape           # class count from the labels


def test_a_dataset_that_contradicts_the_model_spec_is_rejected():
    x, y = _separable_dataset(rows=40, features=6)
    try:
        train({"model_spec": {"architecture": "mlp", "input_dim": 99},
               "hyperparameters": {"steps": 1, "batch_size": 8}},
              lambda _m: None, plan=None, dataset=(x, y))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "input_dim" in str(e), e


def test_wrongly_shaped_features_are_rejected():
    import numpy as np
    try:
        train({"model_spec": {"architecture": "mlp"},
               "hyperparameters": {"steps": 1, "batch_size": 4}},
              lambda _m: None, plan=None,
              dataset=(np.zeros((10, 2, 2), dtype="float32"), np.zeros(10, dtype="int64")))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "2-D" in str(e), e


def test_dataset_survives_the_wire_format_and_still_trains():
    """End to end: pack as it would be sent, unpack as the node would, train."""
    from backend.service.artifacts import pack_dataset, unpack_dataset

    x, y = _separable_dataset(rows=200)
    received_x, received_y = unpack_dataset(pack_dataset(x, y))

    torch.manual_seed(0)
    metrics = train({"model_spec": {"architecture": "mlp", "hidden_dim": 64, "depth": 2},
                     "hyperparameters": {"steps": 50, "batch_size": 64, "learning_rate": 1e-2}},
                    lambda _m: None, plan=None,
                    dataset=(received_x, received_y))["metrics"]

    assert metrics["dataset_rows"] == 200
    assert metrics["final_loss"] < 0.5, metrics


def test_weights_come_back_in_the_safe_wire_format():
    from backend.service.artifacts import pack_state_dict, unpack_state_dict

    x, y = _separable_dataset(rows=60)
    result = train({"model_spec": {"architecture": "mlp", "hidden_dim": 16, "depth": 1},
                    "hyperparameters": {"steps": 2, "batch_size": 16}},
                   lambda _m: None, plan=None, dataset=(x, y))

    restored = unpack_state_dict(pack_state_dict(result["state_dict"]))
    assert set(restored) == set(result["state_dict"])


def test_synthetic_runs_are_flagged_as_such():
    metrics = train({"model_spec": MLP_SPEC,
                     "hyperparameters": {"steps": 1, "batch_size": 8}},
                    lambda _m: None, plan=None)["metrics"]
    assert metrics["synthetic_data"] is True
    assert metrics["dataset_rows"] is None


# --- device resolution ---------------------------------------------------

def test_devices_fall_back_to_cpu_without_a_gpu():
    logs = []
    devices = resolve_devices([], logs.append)
    assert len(devices) == 1 and devices[0].type == "cpu"
    assert any("CPU" in line for line in logs)


# --- end to end ----------------------------------------------------------

def test_train_entry_point_returns_reportable_metrics():
    logs = []
    result = train(
        {"model_name": "demo", "model_spec": MLP_SPEC,
         "hyperparameters": {"steps": 3, "batch_size": 16, "seed": 1}},
        logs.append,
        plan=None,
    )
    metrics = result["metrics"]
    assert result["state_dict"], "trained weights must come back for the submitter"

    for key in ("steps", "batch_size", "parameters", "devices", "seconds",
                "samples_per_second", "achieved_tflops", "achieved_gflops",
                "initial_loss", "final_loss"):
        assert key in metrics, key

    assert metrics["steps"] == 3
    assert metrics["samples_per_second"] > 0
    # A toy model does very little work, so check the unrounded-enough figure.
    assert metrics["achieved_gflops"] > 0, metrics["achieved_gflops"]
    assert logs


def test_train_is_reproducible_for_a_fixed_seed():
    def run():
        return train(
            {"model_spec": MLP_SPEC,
             "hyperparameters": {"steps": 3, "batch_size": 16, "seed": 42}},
            lambda _m: None, plan=None,
        )["metrics"]["final_loss"]

    torch.manual_seed(0)
    first = run()
    torch.manual_seed(0)
    second = run()
    assert abs(first - second) < 1e-6, (first, second)


# --- standalone runner ---------------------------------------------------

def _main():
    if not HAVE_TORCH:
        print("  SKIP  torch is not installed - trainer tests not run")
        return 0

    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as e:
            failed.append(name)
            print("  FAIL  %s: %s" % (name, e))
        except Exception as e:
            failed.append(name)
            print("  ERROR %s: %s: %s" % (name, type(e).__name__, e))
    print("")
    summary = "%d/%d passed" % (len(tests) - len(failed), len(tests))
    if failed:
        summary += " -- FAILED: %s" % ", ".join(failed)
    print(summary)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())
