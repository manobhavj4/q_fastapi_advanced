import yaml
import ssl
import time
import json
import threading
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Callable
import paho.mqtt.client as mqtt


class MQTTClient:
    """Optimized MQTT client for Raspberry Pi with improved error handling and performance."""
    
    def __init__(self, config_path: str = "config/comms_config.yaml"):
        self.config = self._load_config(config_path)
        self.client = None
        self.connected = threading.Event()
        self.shutdown_event = threading.Event()
        self.connection_thread = None
        self.message_callbacks = {}
        self.retry_count = 0
        self.max_retries = self.config.get("max_retries", 10)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize client
        self._initialize_client()
        
        # Configuration
        self.broker = self.config["broker"]
        self.port = self.config.get("port", 8883)
        self.topic_pub = self.config["topic_pub"]
        self.topic_sub = self.config.get("topic_sub")
        self.keepalive = self.config.get("keepalive", 60)
        self.qos = self.config.get("qos", 1)
        
        # Start connection in background
        self._start_connection()

    def _setup_logging(self):
        """Setup logging configuration."""
       

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration with error handling."""
 

    def _initialize_client(self):
        """Initialize MQTT client with optimized settings."""
   
    def _configure_tls(self):
        """Configure TLS with enhanced security."""
      

    def _start_connection(self):
        """Start connection in background thread."""
     

    def _connection_loop(self):
        """Main connection loop with exponential backoff."""
        

    def _handle_connection_failure(self):
        """Handle connection failures with exponential backoff."""
   

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Enhanced connection handler."""
       

    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Enhanced disconnection handler."""
        

    def _on_message(self, client, userdata, msg):
        """Enhanced message handler with callbacks."""
 

    def _on_publish(self, client, userdata, mid):
        """Publish confirmation handler."""
       

    def _on_subscribe(self, client, userdata, mid, granted_qos, properties=None):
        """Subscription confirmation handler."""
     

    def _on_log(self, client, userdata, level, buf):
        """MQTT client logging handler."""
     

    def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """Wait for connection to be established."""
        

    def subscribe(self, topic: str, qos: Optional[int] = None, callback: Optional[Callable] = None):
        """Subscribe to topic with optional callback."""


    def publish(self, payload: Any, topic: Optional[str] = None, qos: Optional[int] = None, retain: bool = False) -> bool:
        """Publish message with improved error handling."""
       

    def unsubscribe(self, topic: str):
        """Unsubscribe from topic."""
      
    def is_connected(self) -> bool:
        """Check if client is connected."""
      

    def disconnect(self):
        """Gracefully disconnect from broker."""
  

    def __enter__(self):
        """Context manager entry."""
      

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
   


# Usage example
if __name__ == "__main__":
    # Example usage with context manager
    try:
        with MQTTClient("config/comms_config.yaml") as mqtt_client:
            # Wait for connection
            if mqtt_client.wait_for_connection():
                print("Connected successfully!")
                
                # Subscribe with callback
                def message_handler(topic, payload):
                    print(f"Received: {payload} on {topic}")
                
                mqtt_client.subscribe("test/topic", callback=message_handler)
                
                # Publish some messages
                mqtt_client.publish({"message": "Hello from RPi!"})
                mqtt_client.publish("Simple string message")
                
                # Keep running
                time.sleep(60)
            else:
                print("Failed to connect")
                
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")