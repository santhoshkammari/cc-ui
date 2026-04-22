"""
Monitor service — health checks, system metrics, provider status.

Tracks:
- Provider availability (which AI backends are online)
- System resources (CPU, memory, disk)
- Task statistics (running, completed, error rates)
- Uptime and response times
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any

log = logging.getLogger("cc-ui.monitor")


class Monitor:
    def __init__(self):
        self._start_time = time.time()
        self._task_stats = {
            "total": 0, "running": 0, "done": 0,
            "error": 0, "stopped": 0,
        }
        self._provider_status: dict[str, dict] = {}
        self._last_health_check: str = ""

    def update_task_stats(self, tasks: dict):
        """Recalculate task statistics from task dict."""
        stats = {"total": 0, "running": 0, "done": 0, "error": 0, "stopped": 0}
        for t in tasks.values():
            stats["total"] += 1
            status = t.get("status", "unknown")
            if status in stats:
                stats[status] += 1
        self._task_stats = stats

    async def check_providers(self) -> dict:
        """Run health checks on all registered providers."""
        try:
            from services.providers.registry import health_check_all
            self._provider_status = await health_check_all()
        except Exception as e:
            self._provider_status = {"error": str(e)}
        self._last_health_check = datetime.now().isoformat()
        return self._provider_status

    def get_system_info(self) -> dict:
        """Get system resource info."""
        info = {
            "uptime_seconds": round(time.time() - self._start_time),
            "pid": os.getpid(),
            "python_version": os.sys.version.split()[0],
        }

        try:
            import psutil
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            info["memory_used_gb"] = round(mem.used / (1024**3), 2)
            info["memory_total_gb"] = round(mem.total / (1024**3), 2)
            info["memory_percent"] = mem.percent
            disk = psutil.disk_usage("/")
            info["disk_used_gb"] = round(disk.used / (1024**3), 2)
            info["disk_total_gb"] = round(disk.total / (1024**3), 2)
        except ImportError:
            pass

        return info

    def get_dashboard(self) -> dict:
        """Full monitoring dashboard data."""
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(time.time() - self._start_time),
            "tasks": self._task_stats,
            "providers": self._provider_status,
            "last_health_check": self._last_health_check,
            "system": self.get_system_info(),
        }

    def get_metrics(self) -> dict:
        """Prometheus-style metrics."""
        m = {
            "ccui_uptime_seconds": round(time.time() - self._start_time),
            "ccui_tasks_total": self._task_stats["total"],
            "ccui_tasks_running": self._task_stats["running"],
            "ccui_tasks_done": self._task_stats["done"],
            "ccui_tasks_error": self._task_stats["error"],
        }
        for name, status in self._provider_status.items():
            m[f"ccui_provider_{name}_up"] = 1 if status.get("status") == "ok" else 0
        return m
