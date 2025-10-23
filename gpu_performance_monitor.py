"""
📊 GPU PERFORMANCE MONITORING SYSTEM
=====================================

Überwacht und tracked:
- GPU-Auslastung (Utilization)
- VRAM-Nutzung
- Temperatur
- Power Draw
- Training-Geschwindigkeit
- Model-Performance-Metriken
- System-Health

Für Windows Server + RTX 3090 optimiert
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import time
import platform
import warnings
warnings.filterwarnings('ignore')

# Versuche nvidia-ml-py3 zu importieren (für detaillierte GPU Stats)
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("⚠️  pynvml nicht installiert - Limitiertes GPU-Monitoring")
    print("   Install: pip install nvidia-ml-py3")


# ==========================================================
# GPU METRICS
# ==========================================================
@dataclass
class GPUMetrics:
    """GPU-Metriken zu einem Zeitpunkt"""
    timestamp: datetime

    # GPU Info
    gpu_name: str
    gpu_index: int = 0

    # Utilization
    gpu_utilization: float = 0.0  # %
    memory_utilization: float = 0.0  # %

    # Memory
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    memory_total_mb: float = 0.0

    # Temperature & Power
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0

    # Performance
    sm_clock_mhz: float = 0.0  # Streaming Multiprocessor Clock
    mem_clock_mhz: float = 0.0  # Memory Clock

    def to_dict(self) -> Dict:
        """Konvertiere zu Dict"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'GPUMetrics':
        """Erstelle aus Dict"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


# ==========================================================
# TRAINING METRICS
# ==========================================================
@dataclass
class TrainingMetrics:
    """Training-Performance-Metriken"""
    timestamp: datetime

    # Training Info
    model_name: str
    epoch: int
    batch_size: int

    # Performance
    samples_per_second: float = 0.0
    avg_batch_time_ms: float = 0.0

    # Loss & Accuracy
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0

    # Learning Rate
    learning_rate: float = 0.0

    # GPU während Training
    gpu_utilization: float = 0.0
    vram_usage_mb: float = 0.0

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


# ==========================================================
# GPU MONITOR
# ==========================================================
class GPUMonitor:
    """
    Echtzeit GPU-Monitoring

    Nutzt:
    - PyTorch CUDA Stats (immer verfügbar)
    - NVML (nvidia-ml-py3) für detaillierte Stats
    """

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

        # NVML Init
        self.nvml_available = False
        if NVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self.nvml_available = True
                print("✅ NVML GPU-Monitoring aktiviert")
            except:
                print("⚠️  NVML Init fehlgeschlagen - Basis-Monitoring")

        # GPU Info
        self.gpu_name = self._get_gpu_name()
        self.gpu_compute_capability = self._get_compute_capability()

        print(f"\n🚀 GPU Monitor initialisiert:")
        print(f"   Device: {self.gpu_name}")
        print(f"   Compute Capability: {self.gpu_compute_capability}")
        print(f"   NVML: {'✅' if self.nvml_available else '❌'}")

    def _get_gpu_name(self) -> str:
        """Hole GPU Name"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(self.gpu_index)
        return "No GPU"

    def _get_compute_capability(self) -> str:
        """Hole Compute Capability"""
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self.gpu_index)
            return f"{capability[0]}.{capability[1]}"
        return "N/A"

    def get_current_metrics(self) -> GPUMetrics:
        """Hole aktuelle GPU-Metriken"""
        metrics = GPUMetrics(
            timestamp=datetime.now(),
            gpu_name=self.gpu_name,
            gpu_index=self.gpu_index
        )

        if not torch.cuda.is_available():
            return metrics

        # PyTorch CUDA Stats (immer verfügbar)
        metrics.memory_allocated_mb = torch.cuda.memory_allocated(self.gpu_index) / 1e6
        metrics.memory_reserved_mb = torch.cuda.memory_reserved(self.gpu_index) / 1e6
        metrics.memory_total_mb = torch.cuda.get_device_properties(self.gpu_index).total_memory / 1e6

        metrics.memory_utilization = (metrics.memory_allocated_mb / metrics.memory_total_mb) * 100

        # NVML Stats (detailliert)
        if self.nvml_available:
            try:
                # Utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                metrics.gpu_utilization = utilization.gpu
                # memory utilization bereits von PyTorch

                # Temperature
                metrics.temperature_c = pynvml.nvmlDeviceGetTemperature(
                    self.handle,
                    pynvml.NVML_TEMPERATURE_GPU
                )

                # Power
                metrics.power_draw_w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                metrics.power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0

                # Clocks
                metrics.sm_clock_mhz = pynvml.nvmlDeviceGetClockInfo(
                    self.handle,
                    pynvml.NVML_CLOCK_SM
                )
                metrics.mem_clock_mhz = pynvml.nvmlDeviceGetClockInfo(
                    self.handle,
                    pynvml.NVML_CLOCK_MEM
                )

            except Exception as e:
                print(f"⚠️  NVML Error: {e}")

        return metrics

    def print_metrics(self, metrics: GPUMetrics = None):
        """Drucke Metriken"""
        if metrics is None:
            metrics = self.get_current_metrics()

        print(f"\n📊 GPU METRICS ({metrics.timestamp.strftime('%H:%M:%S')})")
        print("="*60)
        print(f"GPU: {metrics.gpu_name}")
        print(f"Utilization: {metrics.gpu_utilization:.1f}%")
        print(f"Temperature: {metrics.temperature_c:.1f}°C")
        print(f"Power: {metrics.power_draw_w:.1f}W / {metrics.power_limit_w:.1f}W")
        print(f"\nMemory:")
        print(f"  Allocated: {metrics.memory_allocated_mb:.0f} MB")
        print(f"  Reserved:  {metrics.memory_reserved_mb:.0f} MB")
        print(f"  Total:     {metrics.memory_total_mb:.0f} MB")
        print(f"  Usage:     {metrics.memory_utilization:.1f}%")
        print(f"\nClocks:")
        print(f"  GPU:    {metrics.sm_clock_mhz:.0f} MHz")
        print(f"  Memory: {metrics.mem_clock_mhz:.0f} MHz")
        print("="*60)

    def check_health(self, metrics: GPUMetrics = None) -> Dict[str, bool]:
        """
        Prüfe GPU Health

        Returns:
            Dict mit Health-Checks
        """
        if metrics is None:
            metrics = self.get_current_metrics()

        health = {
            'temperature_ok': metrics.temperature_c < 85.0,  # RTX 3090 Max ~95°C
            'power_ok': metrics.power_draw_w < metrics.power_limit_w * 0.95,
            'memory_ok': metrics.memory_utilization < 95.0,
            'overall_ok': True
        }

        health['overall_ok'] = all([
            health['temperature_ok'],
            health['power_ok'],
            health['memory_ok']
        ])

        return health

    def __del__(self):
        """Cleanup"""
        if self.nvml_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


# ==========================================================
# PERFORMANCE TRACKER
# ==========================================================
class PerformanceTracker:
    """
    Tracked Training-Performance über Zeit

    Speichert alle Metriken für:
    - Langzeit-Analyse
    - Performance-Regression Detection
    - Optimierungs-Opportunities
    """

    def __init__(self, save_dir: str = "logs/performance"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gpu_metrics_history: List[GPUMetrics] = []
        self.training_metrics_history: List[TrainingMetrics] = []

        self.gpu_monitor = GPUMonitor() if torch.cuda.is_available() else None

    def log_gpu_metrics(self):
        """Logge aktuelle GPU-Metriken"""
        if self.gpu_monitor:
            metrics = self.gpu_monitor.get_current_metrics()
            self.gpu_metrics_history.append(metrics)

    def log_training_metrics(
        self,
        model_name: str,
        epoch: int,
        batch_size: int,
        samples_per_second: float,
        avg_batch_time_ms: float,
        train_loss: float,
        val_loss: float,
        train_accuracy: float,
        val_accuracy: float,
        learning_rate: float
    ):
        """Logge Training-Metriken"""
        # GPU Stats
        gpu_util = 0.0
        vram_usage = 0.0

        if self.gpu_monitor:
            gpu_metrics = self.gpu_monitor.get_current_metrics()
            gpu_util = gpu_metrics.gpu_utilization
            vram_usage = gpu_metrics.memory_allocated_mb

        metrics = TrainingMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            epoch=epoch,
            batch_size=batch_size,
            samples_per_second=samples_per_second,
            avg_batch_time_ms=avg_batch_time_ms,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            learning_rate=learning_rate,
            gpu_utilization=gpu_util,
            vram_usage_mb=vram_usage
        )

        self.training_metrics_history.append(metrics)

    def save_logs(self):
        """Speichere alle Logs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # GPU Metrics
        if self.gpu_metrics_history:
            gpu_data = [m.to_dict() for m in self.gpu_metrics_history]
            gpu_df = pd.DataFrame(gpu_data)
            gpu_df.to_csv(self.save_dir / f"gpu_metrics_{timestamp}.csv", index=False)
            print(f"✅ GPU Metrics gespeichert: {len(gpu_data)} Einträge")

        # Training Metrics
        if self.training_metrics_history:
            train_data = [m.to_dict() for m in self.training_metrics_history]
            train_df = pd.DataFrame(train_data)
            train_df.to_csv(self.save_dir / f"training_metrics_{timestamp}.csv", index=False)
            print(f"✅ Training Metrics gespeichert: {len(train_data)} Einträge")

    def get_summary_statistics(self) -> Dict:
        """Berechne Summary Statistics"""
        summary = {}

        # GPU Stats
        if self.gpu_metrics_history:
            temps = [m.temperature_c for m in self.gpu_metrics_history if m.temperature_c > 0]
            power = [m.power_draw_w for m in self.gpu_metrics_history if m.power_draw_w > 0]
            util = [m.gpu_utilization for m in self.gpu_metrics_history if m.gpu_utilization > 0]
            mem = [m.memory_utilization for m in self.gpu_metrics_history if m.memory_utilization > 0]

            summary['gpu'] = {
                'avg_temperature': np.mean(temps) if temps else 0,
                'max_temperature': np.max(temps) if temps else 0,
                'avg_power': np.mean(power) if power else 0,
                'max_power': np.max(power) if power else 0,
                'avg_utilization': np.mean(util) if util else 0,
                'avg_memory_usage': np.mean(mem) if mem else 0
            }

        # Training Stats
        if self.training_metrics_history:
            sps = [m.samples_per_second for m in self.training_metrics_history if m.samples_per_second > 0]

            summary['training'] = {
                'avg_samples_per_second': np.mean(sps) if sps else 0,
                'total_epochs': len(self.training_metrics_history)
            }

        return summary

    def print_summary(self):
        """Drucke Summary"""
        summary = self.get_summary_statistics()

        print("\n📊 PERFORMANCE SUMMARY")
        print("="*60)

        if 'gpu' in summary:
            gpu = summary['gpu']
            print("GPU:")
            print(f"  Avg Temperature: {gpu['avg_temperature']:.1f}°C (Max: {gpu['max_temperature']:.1f}°C)")
            print(f"  Avg Power:       {gpu['avg_power']:.1f}W (Max: {gpu['max_power']:.1f}W)")
            print(f"  Avg Utilization: {gpu['avg_utilization']:.1f}%")
            print(f"  Avg Memory:      {gpu['avg_memory_usage']:.1f}%")

        if 'training' in summary:
            train = summary['training']
            print("\nTraining:")
            print(f"  Avg Throughput:  {train['avg_samples_per_second']:.0f} samples/sec")
            print(f"  Total Epochs:    {train['total_epochs']}")

        print("="*60)


# ==========================================================
# CONTINUOUS MONITORING
# ==========================================================
class ContinuousMonitor:
    """
    Continuous Background Monitoring

    Läuft im Hintergrund und loggt Metriken
    """

    def __init__(self, log_interval_seconds: int = 60):
        self.log_interval = log_interval_seconds
        self.tracker = PerformanceTracker()
        self.is_running = False

    def start(self):
        """Starte Monitoring"""
        print(f"🔄 Continuous Monitoring gestartet (Interval: {self.log_interval}s)")

        self.is_running = True

        while self.is_running:
            try:
                # Log GPU Metrics
                self.tracker.log_gpu_metrics()

                # Wait
                time.sleep(self.log_interval)

            except KeyboardInterrupt:
                print("\n⏹️  Monitoring gestoppt")
                self.is_running = False
                break

        # Save on Exit
        self.tracker.save_logs()
        self.tracker.print_summary()

    def stop(self):
        """Stoppe Monitoring"""
        self.is_running = False


# ==========================================================
# EXAMPLE USAGE
# ==========================================================
if __name__ == "__main__":
    print("📊 GPU PERFORMANCE MONITOR TEST")
    print("="*60)

    # GPU Monitor
    if torch.cuda.is_available():
        monitor = GPUMonitor()

        # Current Metrics
        metrics = monitor.get_current_metrics()
        monitor.print_metrics(metrics)

        # Health Check
        health = monitor.check_health(metrics)
        print(f"\n🏥 Health Check:")
        for check, status in health.items():
            print(f"   {check}: {'✅' if status else '❌'}")

        # Performance Tracker
        print("\n📈 Testing Performance Tracker...")
        tracker = PerformanceTracker()

        # Simuliere Training
        for epoch in range(5):
            tracker.log_training_metrics(
                model_name="TestModel",
                epoch=epoch,
                batch_size=256,
                samples_per_second=1500.0,
                avg_batch_time_ms=2.5,
                train_loss=0.5 - epoch * 0.05,
                val_loss=0.55 - epoch * 0.04,
                train_accuracy=0.7 + epoch * 0.03,
                val_accuracy=0.68 + epoch * 0.03,
                learning_rate=0.001
            )

            tracker.log_gpu_metrics()
            time.sleep(0.1)

        # Summary
        tracker.print_summary()

        # Save
        tracker.save_logs()

    else:
        print("❌ Keine GPU verfügbar - Monitoring nicht möglich")

    print("\n✅ Test abgeschlossen!")
