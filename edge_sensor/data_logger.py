import os
import json
import csv
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path
from contextlib import contextmanager
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from botocore.config import Config

# Optional: InfluxDB client (if used)
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import ASYNCHRONOUS
    INFLUX_AVAILABLE = True
except ImportError:
    INFLUX_AVAILABLE = False

# Configure logging



class DataLogger:
    """
    Optimized data logger for Raspberry Pi with:
    - Batch processing for improved performance
    - Async operations for non-blocking I/O
    - Better error handling and retry logic
    - Resource management with context managers
    - Thread-safe operations
    """
    
    def __init__(self, config: Dict[str, Any]):
 
        
    def _setup_s3(self):
        """Initialize S3 client with optimized configuration"""
      
        
    def _setup_influxdb(self):
        """Initialize InfluxDB client"""
    
        
    @contextmanager
    def _get_csv_writer(self, filename: str, fieldnames: List[str]):
        """Context manager for CSV file operations with caching"""
     
            
    def _batch_processor(self):
        """Background thread for batch processing data"""
      
            
    def _process_batch(self, batch_data: List[Dict[str, Any]]):
        """Process a batch of data entries"""
        
            
    def _write_batch_to_files(self, data_list: List[Dict[str, Any]], date_str: str):
        """Write batch data to local files"""
        
            
    def _batch_influxdb_write(self, batch_data: List[Dict[str, Any]]):
        """Write batch data to InfluxDB"""

            
    def log(self, data: Dict[str, Any]):
        """
        Add data to the processing queue (non-blocking)
        """
    
            
    def upload_to_s3(self, local_file: str, retries: int = 3):
        """
        Upload file to S3 with retry logic
        """
       
        
    def sync_to_s3(self, pattern: str = "*.csv"):
        """
        Sync local files to S3 (useful for periodic uploads)
        """
       
                    
    def flush(self):
        """Force flush of all pending data"""
        # Wait for queue to empty
    
            
    def close(self):
        """Clean shutdown of the logger"""
       
        
    def __enter__(self):
  
        
    def __exit__(self, exc_type, exc_val, exc_tb):
      

# Example usage with context manager
if __name__ == "__main__":
    # Example config (could be loaded from YAML or .env)
    config = {
        "local_dir": "./logs",
        "s3_enabled": True,
        "s3_bucket": "my-qusp-bucket",
        "s3_path": "edge_logs/",
        "influx_enabled": False,
        "batch_size": 50,
        "flush_interval": 30
    }

    # Use context manager for proper cleanup
    with DataLogger(config) as logger_instance:
        # Simulate sensor data
        for i in range(100):
            dummy_data = {
                "device_id": "edge01",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "temperature": 23.7 + (i * 0.1),
                "vibration_rms": 0.0051 + (i * 0.0001),
                "quantum_signal": 0.8214 + (i * 0.001)
            }
            
            logger_instance.log(dummy_data)
            time.sleep(0.1)  # Simulate sensor reading interval
            
        # Ensure all data is processed
        logger_instance.flush()
        
        # Periodic S3 sync
        logger_instance.sync_to_s3()