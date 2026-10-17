import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

from mamba2_lob import build_engineered_features_numpy, inverse_target_transform
from utils import DataPoint, ScorerStepByStep


@dataclass
class SequenceState:
    hidden_states: List[np.ndarray]
    prev_raw: Optional[np.ndarray]


class PredictionModel:
    """
    Stateful Mamba2-style ONNX inference model.

    Expected files in the same directory:
      - mamba2_step.onnx
      - model_meta.json
    """

    def __init__(self, model_path: str = ""):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = model_path or os.path.join(base_dir, "mamba2_step.onnx")
        meta_path = os.path.join(base_dir, "model_meta.json")

        self.sessions_ready = False
        self.state_cache: Dict[int, SequenceState] = {}

        self.n_layers = 4
        self.d_model = 128
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.n_layers = int(meta.get("n_layers", self.n_layers))
            self.d_model = int(meta.get("d_model", self.d_model))

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.ort_session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self.output_names = [out.name for out in self.ort_session.get_outputs()]
            self.input_names = [inp.name for inp in self.ort_session.get_inputs()]
            self.sessions_ready = True
            print(f"Loaded model: {onnx_path}")
        except Exception as exc:
            print(f"Failed to load ONNX model: {exc}")
            self.ort_session = None

    def _new_sequence_state(self) -> SequenceState:
        hidden = [np.zeros((1, self.d_model), dtype=np.float32) for _ in range(self.n_layers)]
        return SequenceState(hidden_states=hidden, prev_raw=None)

    def _get_state(self, seq_ix: int, step_in_seq: int) -> SequenceState:
        if step_in_seq == 0 or seq_ix not in self.state_cache:
            self.state_cache[seq_ix] = self._new_sequence_state()
        return self.state_cache[seq_ix]

    def predict(self, data_point: DataPoint) -> Optional[np.ndarray]:
        if not self.sessions_ready:
            return None if not data_point.need_prediction else np.zeros(2, dtype=np.float32)

        seq_ix = int(data_point.seq_ix)
        step_in_seq = int(data_point.step_in_seq)

        seq_state = self._get_state(seq_ix, step_in_seq)

        raw = np.asarray(data_point.state, dtype=np.float32)
        if seq_state.prev_raw is None:
            delta = np.zeros_like(raw, dtype=np.float32)
        else:
            delta = raw - seq_state.prev_raw

        feature = build_engineered_features_numpy(raw=raw, delta=delta, step_in_seq=step_in_seq)
        seq_state.prev_raw = raw.copy()

        ort_inputs = {"feature": feature.reshape(1, -1).astype(np.float32)}
        for i in range(self.n_layers):
            ort_inputs[f"state_{i}"] = seq_state.hidden_states[i]

        outputs = self.ort_session.run(self.output_names, ort_inputs)
        pred_transformed = outputs[0][0]

        for i in range(self.n_layers):
            seq_state.hidden_states[i] = outputs[i + 1].astype(np.float32)

        if not data_point.need_prediction:
            return None

        pred = inverse_target_transform(pred_transformed.astype(np.float32))
        return pred.astype(np.float32)


if __name__ == "__main__":
    valid_path = f"{CURRENT_DIR}/../datasets/valid.parquet"
    if not os.path.exists(valid_path):
        print("Validation dataset not found.")
        raise SystemExit(0)

    model = PredictionModel()
    scorer = ScorerStepByStep(valid_path)
    results = scorer.score(model)

    print("Local validation results:")
    print(f"Mean weighted_pearson: {results['weighted_pearson']:.6f}")
    print(f"t0: {results['t0']:.6f}")
    print(f"t1: {results['t1']:.6f}")
