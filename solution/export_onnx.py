import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import warnings
import torch
from mamba2_lob import Mamba2LOBModel, ModelConfig


class StepWrapper(torch.nn.Module):
    def __init__(self, model: Mamba2LOBModel):
        super().__init__()
        self.model = model

    def forward(self, feature: torch.Tensor, *states: torch.Tensor):
        state_list = list(states) if len(states) > 0 else None
        pred, new_states = self.model.step(feature, state_list)
        return (pred, *new_states)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Mamba2 LOB model to ONNX (step mode)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mamba2_lob/best.pt")
    parser.add_argument("--out-dir", type=str, default="solution")
    parser.add_argument("--onnx-name", type=str, default="mamba2_step.onnx")
    parser.add_argument("--meta-name", type=str, default="model_meta.json")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--exporter",
        type=str,
        default="auto",
        choices=["auto", "dynamo", "legacy"],
        help="ONNX exporter backend selection.",
    )
    return parser.parse_args()


def _supports_dynamo_exporter() -> bool:
    params = inspect.signature(torch.onnx.export).parameters
    return "dynamo" in params


def _supports_dynamic_shapes() -> bool:
    params = inspect.signature(torch.onnx.export).parameters
    return "dynamic_shapes" in params


def _build_dynamic_shapes(n_layers: int) -> dict | None:
    torch_export = getattr(torch, "export", None)
    dim_ctor = getattr(torch_export, "Dim", None)
    if dim_ctor is None:
        return None

    batch = dim_ctor("batch")
    dynamic_shapes = {"feature": {0: batch}}
    for i in range(n_layers):
        dynamic_shapes[f"state_{i}"] = {0: batch}
    return dynamic_shapes


def _has_onnxscript() -> bool:
    return importlib.util.find_spec("onnxscript") is not None


def _run_export(
    wrapper: torch.nn.Module,
    feature: torch.Tensor,
    states: list[torch.Tensor],
    onnx_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict,
    opset: int,
    use_dynamo: bool,
) -> None:
    kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )
    if _supports_dynamo_exporter():
        kwargs["dynamo"] = use_dynamo

    if use_dynamo:
        if _supports_dynamic_shapes():
            dynamic_shapes = _build_dynamic_shapes(n_layers=len(states))
            if dynamic_shapes is not None:
                kwargs["dynamic_shapes"] = dynamic_shapes
        torch.onnx.export(wrapper, (feature, *states), onnx_path.as_posix(), **kwargs)
        return

    kwargs["dynamic_axes"] = dynamic_axes
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*legacy TorchScript-based ONNX export.*",
        )
        torch.onnx.export(wrapper, (feature, *states), onnx_path.as_posix(), **kwargs)


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ModelConfig(**ckpt["config"])

    model = Mamba2LOBModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = StepWrapper(model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature = torch.zeros(1, config.input_dim, dtype=torch.float32)
    states = [torch.zeros(1, config.d_model, dtype=torch.float32) for _ in range(config.n_layers)]

    input_names = ["feature"] + [f"state_{i}" for i in range(config.n_layers)]
    output_names = ["pred"] + [f"new_state_{i}" for i in range(config.n_layers)]

    dynamic_axes = {"feature": {0: "batch"}, "pred": {0: "batch"}}
    for i in range(config.n_layers):
        dynamic_axes[f"state_{i}"] = {0: "batch"}
        dynamic_axes[f"new_state_{i}"] = {0: "batch"}

    onnx_path = out_dir / args.onnx_name
    has_dynamo = _supports_dynamo_exporter()
    if args.exporter == "legacy":
        _run_export(
            wrapper=wrapper,
            feature=feature,
            states=states,
            onnx_path=onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset=args.opset,
            use_dynamo=False,
        )
        print("ONNX export backend: legacy")
    elif args.exporter == "dynamo":
        if not has_dynamo:
            raise RuntimeError("This torch version does not support torch.onnx.export(..., dynamo=True).")
        _run_export(
            wrapper=wrapper,
            feature=feature,
            states=states,
            onnx_path=onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset=args.opset,
            use_dynamo=True,
        )
        print("ONNX export backend: dynamo")
    else:
        if has_dynamo:
            if _has_onnxscript():
                try:
                    _run_export(
                        wrapper=wrapper,
                        feature=feature,
                        states=states,
                        onnx_path=onnx_path,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        opset=args.opset,
                        use_dynamo=True,
                    )
                    print("ONNX export backend: dynamo")
                except Exception as exc:
                    print(f"Dynamo export failed, falling back to legacy exporter. Error: {exc}")
                    _run_export(
                        wrapper=wrapper,
                        feature=feature,
                        states=states,
                        onnx_path=onnx_path,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        opset=args.opset,
                        use_dynamo=False,
                    )
                    print("ONNX export backend: legacy (fallback)")
            else:
                print("onnxscript is not installed; using legacy ONNX exporter.")
                _run_export(
                    wrapper=wrapper,
                    feature=feature,
                    states=states,
                    onnx_path=onnx_path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset=args.opset,
                    use_dynamo=False,
                )
                print("ONNX export backend: legacy")
        else:
            _run_export(
                wrapper=wrapper,
                feature=feature,
                states=states,
                onnx_path=onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset=args.opset,
                use_dynamo=False,
            )
            print("ONNX export backend: legacy")

    meta = {
        "input_dim": config.input_dim,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "target_transform": "sign_log1p",
        "checkpoint": args.checkpoint,
        "onnx_model": args.onnx_name,
    }

    with (out_dir / args.meta_name).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported ONNX model to {onnx_path}")
    print(f"Saved metadata to {out_dir / args.meta_name}")


if __name__ == "__main__":
    main()
