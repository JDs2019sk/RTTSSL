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

class PerformanceMonitor:
    def __init__(self, history_size=100):
        self.fps_history = deque(maxlen=history_size)
        self.processing_times = deque(maxlen=history_size)
        self.memory_usage = deque(maxlen=history_size)
        self.cpu_usage = deque(maxlen=history_size)
        
        self.start_time = None
        self.frame_count = 0
        self.process = psutil.Process()
        
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
        return {
            'fps': np.mean(self.fps_history) if self.fps_history else 0,
            'processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
            'cpu_usage': np.mean(self.cpu_usage) if self.cpu_usage else 0
        }
        
    def _monitor_system(self):
        """Monitor system resources in background thread"""
        while self.monitoring:
            try:
                self.memory_usage.append(self.process.memory_percent())
                self.cpu_usage.append(psutil.cpu_percent())
            except:
                pass
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
        return {
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
