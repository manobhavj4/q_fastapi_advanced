import unittest
import numpy as np
from signal_processor import SignalProcessor


class TestSignalProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):


    def setUp(self):
        """Create processor instance for each test."""
    

    def test_process_pipeline(self):
        """Test the full processing pipeline: filter, smooth, normalize, outlier removal."""
      

    def test_fft_output(self):
        """Test that FFT returns valid frequency spectrum."""
   

    def test_lowpass_filter(self):
        """Test low-pass filter preserves signal length and reduces high frequency content."""
      

    def test_normalization_zscore(self):
        """Test z-score normalization produces standard normal distribution."""
 

    def test_windowing_shape(self):
        """Test signal windowing returns correct shape and valid values."""
      

    def test_remove_outliers(self):
        """Test outlier removal effectiveness and data integrity."""


    def test_config_validation(self):
        """Test that processor handles invalid configurations gracefully."""
        # Test with invalid sampling rate
        

    def test_empty_signal_handling(self):
        """Test processor behavior with edge cases."""
 

    def test_single_value_signal(self):
        """Test processor with single-value signal."""
      


if __name__ == "__main__":
    # Run tests with more verbose output
  