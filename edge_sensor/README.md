# 🧠 edge_sensor – Edge Runtime for Quantum Sensing Platform (QuSP)

This module forms the **core runtime environment for the edge device** in the QuSP system. It interfaces with quantum sensors, processes signals in real time, performs AI inference, logs data locally/remotely, and communicates with the cloud (e.g., AWS IoT) via MQTT.

---

## 📦 Directory Structure

edge_sensor/
├── edge_main.py                        # Main entry point to launch the edge device runtime
├── sensor_driver.py               # Interface for physical hardware: ADC, lasers, RF
├── sensor_simulator.py            # Simulates sensor output data for testing
├── __init__.py
├── signal_processor.py            # Processes signals: noise filtering, normalization, etc.
├── ai_models/                     # ML models used at the edge for inference
│   ├── drift_compensation.py      # Corrects signal drift using LSTM/Kalman filters
│   ├── anomaly_detector.py        # Detects abnormal behavior or faults
│   ├── fft_feature_extractor.py   # Converts time domain data to frequency features
│   └── model_utils.py             # Loads/saves ONNX or TFLite models
├── data_logger.py                 # Logs sensor data locally or remotely
├── mqtt_client.py                 # Sends data to cloud over MQTT (e.g., AWS IoT)
├── comms_config_loader.py
├── config_loader.py
├── config/                        # Configurations for sensors and communication
│   ├── sensor_config.yaml         # Sampling rate, gain, calibration info
│   └── comms_config.yaml          # MQTT, endpoints, S3 targets
├── tests/                         # Unit tests for the edge components
│   ├── test_driver.py
│   ├── test_signal_pipeline.py
│   └── dummy_data.json
└── README.md                      # Documentation for edge sensor subsystem


---

## 🛠️ Requirements

Install the following Python libraries:

```bash
pip install -r requirements.txt


Key Classes & Functions:

1. sensor_driver.py
SensorDriver: Class to interact with ADC, lasers, quantum sensors

driver = SensorDriver()
data = driver.read_sensor()

2. sensor_simulator.py
SensorSimulator: Simulates raw signal from quantum device for dev/testing.

sim = SensorSimulator()
fake_data = sim.generate()

3. signal_processor.py
SignalProcessor: Applies filtering, normalization, feature enhancement

processor = SignalProcessor()
clean_data = processor.process(raw_data)

4. ai_models/drift_compensation.py
DriftCompensator: Applies Kalman or LSTM-based correction

compensator = DriftCompensator()
corrected = compensator.apply(clean_data)

5. ai_models/anomaly_detector.py
AnomalyDetector: Detects faults in signal using a trained ML model

detector = AnomalyDetector()
is_anomaly = detector.predict(features)

6. ai_models/fft_feature_extractor.py
FFTFeatureExtractor: Transforms time-domain to frequency-domain

extractor = FFTFeatureExtractor()
freq_features = extractor.extract(clean_data)

7. ai_models/model_utils.py
load_model(model_path): Loads ONNX or TFLite models
run_inference(model, data): Performs inference on new data

8. mqtt_client.py
MQTTClient: Publishes messages securely to AWS IoT or similar MQTT brokers

client = MQTTClient(config)
client.send(data)

9. data_logger.py
DataLogger: Logs data locally or sends to cloud storage

logger = DataLogger("local_logs/")
logger.log(data)

Running the Edge Application
python edge_main.py


🧪 Run Tests

cd tests
python test_driver.py
python test_signal_pipeline.py