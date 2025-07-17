# 🧠 AI Control Engine for Quantum-Edge Systems

The `ai_control_engine` is the central AI brain of the **Quantum Sensing and Control Platform**. It orchestrates real-time inference, decision-making, feedback control, and coordination across sensor hardware and quantum logic subsystems. The system leverages advanced ML models including LSTMs, Reinforcement Learning (RL), and CNNs, enabling intelligent signal interpretation, drift prediction, and gate optimization.

---

## 📁 Directory Structure


ai_control_engine/
├── __init__.py
├── controller.py                  # AI manager coordinating sensors and quantum logic
├── digital_twin.py                # ML model that mimics hardware behavior for feedback
├── model_manager.py               # Loads/manages trained models (ONNX, PyTorch)
├── orchestrator.py                # Makes real-time decisions across the system
├── config/                      # Local configs for AI engine paths and thresholds
│    ├── config.py
│    ├── ai_engine_config.yaml
├── ai_control_engine_main.py
├── utils.py                       # Shared utilities (timing, JSON ops, etc.)
├── models/                        # ML model definitions used for training/inference
│   ├── __init__.py
│   ├── lstm_drift_predictor.py
│   ├── qubit_rl_tuner.py
│   ├── qec_decoder_cnn.py
│   ├── digital_twin_net.py
│   └── signal_analyzer/
│       └── analyzer_model.py
├── trainers/                      # Training scripts and workflows for core models
│   ├── train_rl_tuner.py
│   ├── train_drift_model.py
│   ├── train_qec_decoder.py
│   └── train_digital_twin.py
├── registry/                      # MLflow or model versioning integrations
│   ├── model_registry.py
│   ├── register_model.py
│   └── load_model.py
├── README.md                      # Documentation for edge sensor subsystem
└── tests/                         # Tests for AI logic
    ├── test_controller.py
    ├── test_models.py
    └── fixtures/
        └── dummy_input_data.json




---

## ⚙️ Core Components

### 🔁 `controller.py`
Manages the lifecycle of sensor reads, AI inference, error correction, and quantum signal control. Acts as the central orchestrator between modules.

### 🧠 `digital_twin.py`
Trains and runs a digital twin model using LSTM or CNN to replicate hardware/sensor behavior and enable feedback control in real-time.

### 🗃️ `model_manager.py`
Handles loading of models in ONNX or PyTorch formats, caching, and model versioning via the registry.

### 🧬 `orchestrator.py`
Implements rules or AI-based logic to decide system actions (e.g., apply gate, tune voltage) based on current predictions and states.

---

## 🧪 ML Models

| Model                      | Purpose                            |
|---------------------------|------------------------------------|
| `lstm_drift_predictor.py` | Predicts qubit drift/decoherence   |
| `qubit_rl_tuner.py`       | RL agent for tuning control params |
| `qec_decoder_cnn.py`      | CNN decoder for quantum errors     |
| `digital_twin_net.py`     | Mimics real sensor hardware        |
| `signal_analyzer/`        | Tools for raw signal interpretation|

All models are trainable via `trainers/` and trackable via `registry/`.

---

## 🔧 Configuration

Stored in `config/ai_engine_config.yaml`, it contains:
```yaml
sampling_interval: 1.0
drift_threshold: 0.85
model_paths:
  drift_model: "models/lstm_drift.onnx"
  qec_decoder: "models/qec_decoder_cnn.pt"

"""

How to Run
bash
# Step 1: Set environment variables
export AI_ENGINE_CONFIG=config/ai_engine_config.yaml

# Step 2: Run the AI engine
python ai_control_engine_main.py

Testing
bash
# Run all unit tests
pytest tests/
Includes fixtures for dummy input testing.

Model Versioning & Registry
All models are version-controlled and tracked using MLflow or custom logic in:
registry/model_registry.py
registry/register_model.py
registry/load_model.py

Dependencies
PyTorch, ONNX, scikit-learn
yaml, json, time, logging
MLflow (optional, for model registry)
pytest for testing

Install with:

bash
pip install -r requirements.txt

Integration Points
Connected to quantum_control/ (pulse shaping, QEC)
Receives sensor data from edge_sensor/
Sends control feedback via MQTT or shared state
Model lifecycle controlled by model_registry


Authors
Narasingh Prasad Joshi – Quantum-AI Systems Lead
Manobhav Joshi – ML Architect, GenXZAI


"""