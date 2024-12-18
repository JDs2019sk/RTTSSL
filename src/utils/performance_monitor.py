"""
Performance Monitor Module
Provides advanced performance monitoring and optimization features.
"""

import time
import psutil
import numpy as np
from collections import deque
import threading
import cv2
try:
    import pynvml
    NVIDIA_GPU_AVAILABLE = True
except ImportError:
    NVIDIA_GPU_AVAILABLE = False

class PerformanceMonitor:
    def __init__(self, history_size=100):
        self.fps_history = deque(maxlen=history_size)
        self.processing_times = deque(maxlen=history_size)
        self.memory_usage = deque(maxlen=history_size)
        self.cpu_usage = deque(maxlen=history_size)
        self.gpu_usage = deque(maxlen=history_size)
        self.gpu_memory = deque(maxlen=history_size)
        
        self.start_time = None
        self.frame_count = 0
        self.process = psutil.Process()
        
        # Initialize NVIDIA GPU monitoring if available
        self.has_gpu = False
        if NVIDIA_GPU_AVAILABLE:
            try:
                print("\nInitializing NVIDIA GPU monitoring...")
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.has_gpu = True
                gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                print(f"Detected GPU: {gpu_name}")
                print(f"Total Memory: {gpu_info.total / 1024**2:.0f}MB")
                print(f"Used Memory: {gpu_info.used / 1024**2:.0f}MB")
                print(f"Free Memory: {gpu_info.free / 1024**2:.0f}MB")
                print("GPU monitoring initialized successfully!")
            except Exception as e:
                print(f"\nError initializing GPU monitoring:")
                print(f"Error details: {str(e)}")
                if "not found" in str(e).lower():
                    print("NVIDIA drivers not found. Please install the latest drivers.")
                elif "no cuda-capable device" in str(e).lower():
                    print("No NVIDIA GPU detected.")
                else:
                    print("Unknown error initializing GPU.")
                self.has_gpu = False
        else:
            print("\nNVIDIA ML Python library not found.")
            print("GPU metrics will not be available.")
            print("Install the library with: pip install nvidia-ml-py3")
        
        # Performance flags
        self.enable_threading = True
        self.enable_gpu = True
        self.optimize_resolution = True
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def start_frame(self):
        """Start timing a new frame"""
        self.start_time = time.time()
        
    def end_frame(self):
        """End frame timing and update metrics"""
        if self.start_time is None:
            return
            
        # Calculate processing time
        processing_time = time.time() - self.start_time
        self.processing_times.append(processing_time)
        
        # Update FPS
        self.frame_count += 1
        self.fps_history.append(1.0 / processing_time if processing_time > 0 else 0)
        
        self.start_time = None
        
    def get_metrics(self):
        """Get current performance metrics"""
        metrics = {
            'fps': np.mean(self.fps_history) if self.fps_history else 0,
            'processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
            'cpu_usage': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'total_frames': self.frame_count,
            'avg_frame_time': 1000 * np.mean(self.processing_times) if self.processing_times else 0,  # in ms
            'min_fps': np.min(self.fps_history) if self.fps_history else 0,
            'max_fps': np.max(self.fps_history) if self.fps_history else 0,
            'memory_total': psutil.virtual_memory().total / (1024**3),  # in GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # in GB
            'cpu_freq': psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), 'current') else 0,
            'cpu_cores': psutil.cpu_count()
        }
        
        if self.has_gpu:
            try:
                gpu_temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert mW to W
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                
                metrics.update({
                    'gpu_usage': np.mean(self.gpu_usage) if self.gpu_usage else 0,
                    'gpu_memory': np.mean(self.gpu_memory) if self.gpu_memory else 0,
                    'gpu_temp': gpu_temp,
                    'gpu_power': gpu_power,
                    'gpu_memory_total': gpu_info.total / (1024**2),  # in MB
                    'gpu_memory_used': gpu_info.used / (1024**2),  # in MB
                    'gpu_memory_free': gpu_info.free / (1024**2)  # in MB
                })
            except Exception as e:
                print(f"Error getting GPU metrics: {e}")
                metrics.update({
                    'gpu_usage': 0,
                    'gpu_memory': 0
                })
        
        return metrics
        
    def _monitor_system(self):
        """Monitor system resources in a separate thread"""
        while self.monitoring:
            try:
                # CPU and Memory monitoring
                self.cpu_usage.append(psutil.cpu_percent())
                self.memory_usage.append(self.process.memory_percent())
                
                # GPU monitoring if available
                if self.has_gpu:
                    try:
                        # GPU utilization
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        self.gpu_usage.append(float(gpu_util.gpu))
                        
                        # GPU memory
                        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        gpu_mem_used_percent = (float(gpu_mem.used) / float(gpu_mem.total)) * 100.0
                        self.gpu_memory.append(gpu_mem_used_percent)
                        
                        # Print debug info every 10 seconds
                        if len(self.gpu_usage) % 10 == 0:
                            print(f"\nGPU Usage: {self.gpu_usage[-1]:.1f}%")
                            print(f"GPU Memory: {self.gpu_memory[-1]:.1f}%")
                    except Exception as e:
                        print(f"\nError monitoring GPU: {str(e)}")
                        self.gpu_usage.append(0.0)
                        self.gpu_memory.append(0.0)
                
                time.sleep(1)  # Update every second
            except Exception as e:
                print(f"Error monitoring system: {e}")
                time.sleep(1)
                
    def optimize_frame(self, frame):
        """Optimize frame based on performance metrics"""
        if not self.optimize_resolution:
            return frame
            
        metrics = self.get_metrics()
        current_fps = metrics['fps']
        target_fps = 30
        
        if current_fps < target_fps * 0.8:  # If FPS is too low
            # Reduce resolution
            height, width = frame.shape[:2]
            scale_factor = max(0.5, min(1.0, current_fps / target_fps))
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height))
            
        return frame
        
    def get_optimization_suggestions(self):
        """Get performance optimization suggestions"""
        metrics = self.get_metrics()
        suggestions = []
        
        if metrics['fps'] < 20:
            suggestions.append("Consider reducing resolution or disabling some features")
        if metrics['memory_usage'] > 80:
            suggestions.append("High memory usage detected. Consider closing other applications")
        if metrics['cpu_usage'] > 90:
            suggestions.append("High CPU usage. Consider enabling GPU acceleration if available")
            
        return suggestions
        
    def cleanup(self):
        """Cleanup monitoring resources"""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join()
            
    def get_performance_report(self):
        """Generate detailed performance report"""
        metrics = self.get_metrics()
        report = {
            'average_fps': metrics['fps'],
            'frame_time': metrics['processing_time'] * 1000,  # ms
            'memory_usage': metrics['memory_usage'],
            'cpu_usage': metrics['cpu_usage'],
            'total_frames': self.frame_count,
            'optimization_status': {
                'threading': self.enable_threading,
                'gpu': self.enable_gpu,
                'resolution': self.optimize_resolution
            }
        }
        
        if self.has_gpu:
            report.update({
                'gpu_usage': metrics['gpu_usage'],
                'gpu_memory': metrics['gpu_memory']
            })
            
        return report
