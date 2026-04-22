"""
Scheduler service — cron jobs, delayed tasks, recurring jobs.

Supports:
- One-time delayed tasks (run prompt after N seconds)
- Recurring cron-style tasks (e.g., "every day at 9am")
- Task queuing with priority
- Persistent scheduling (survives restarts via SQLite)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Callable, Any

log = logging.getLogger("cc-ui.scheduler")


class ScheduledJob:
    def __init__(self, id: str, name: str, prompt: str, model: str = "claude",
                 mode: str = "bypassPermissions", cwd: str = "",
                 schedule: str = "", next_run: str = "",
                 interval_seconds: int = 0, one_shot: bool = False,
                 enabled: bool = True, last_run: str = "",
                 last_status: str = "", created_at: str = ""):
        self.id = id
        self.name = name
        self.prompt = prompt
        self.model = model
        self.mode = mode
        self.cwd = cwd
        self.schedule = schedule  # cron expression or human-readable
        self.next_run = next_run
        self.interval_seconds = interval_seconds
        self.one_shot = one_shot
        self.enabled = enabled
        self.last_run = last_run
        self.last_status = last_status
        self.created_at = created_at or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "prompt": self.prompt,
            "model": self.model, "mode": self.mode, "cwd": self.cwd,
            "schedule": self.schedule, "next_run": self.next_run,
            "interval_seconds": self.interval_seconds, "one_shot": self.one_shot,
            "enabled": self.enabled, "last_run": self.last_run,
            "last_status": self.last_status, "created_at": self.created_at,
        }


class Scheduler:
    def __init__(self, db_path: str, task_callback: Callable = None):
        self.db_path = db_path
        self.task_callback = task_callback  # async fn(prompt, model, mode, cwd) -> task_id
        self._jobs: dict[str, ScheduledJob] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._init_db()
        self._load_jobs()

    def _db(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._db() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS scheduled_jobs (
                id TEXT PRIMARY KEY,
                name TEXT, prompt TEXT, model TEXT, mode TEXT, cwd TEXT,
                schedule TEXT, next_run TEXT, interval_seconds INTEGER,
                one_shot INTEGER, enabled INTEGER,
                last_run TEXT, last_status TEXT, created_at TEXT
            )""")

    def _load_jobs(self):
        with self._db() as con:
            rows = con.execute("SELECT * FROM scheduled_jobs").fetchall()
        for r in rows:
            job = ScheduledJob(
                id=r[0], name=r[1], prompt=r[2], model=r[3], mode=r[4],
                cwd=r[5], schedule=r[6], next_run=r[7],
                interval_seconds=r[8], one_shot=bool(r[9]),
                enabled=bool(r[10]), last_run=r[11], last_status=r[12],
                created_at=r[13],
            )
            self._jobs[job.id] = job

    def _save_job(self, job: ScheduledJob):
        with self._db() as con:
            con.execute("""INSERT OR REPLACE INTO scheduled_jobs
                (id, name, prompt, model, mode, cwd, schedule, next_run,
                 interval_seconds, one_shot, enabled, last_run, last_status, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (job.id, job.name, job.prompt, job.model, job.mode, job.cwd,
                 job.schedule, job.next_run, job.interval_seconds,
                 int(job.one_shot), int(job.enabled),
                 job.last_run, job.last_status, job.created_at))

    def _delete_job(self, job_id: str):
        with self._db() as con:
            con.execute("DELETE FROM scheduled_jobs WHERE id=?", (job_id,))
        self._jobs.pop(job_id, None)

    def add_job(self, name: str, prompt: str, model: str = "claude",
                mode: str = "bypassPermissions", cwd: str = "",
                schedule: str = "", interval_seconds: int = 0,
                delay_seconds: int = 0, one_shot: bool = False) -> ScheduledJob:
        """Add a new scheduled job."""
        job_id = str(uuid.uuid4())[:8]

        if delay_seconds > 0:
            next_run = (datetime.now() + timedelta(seconds=delay_seconds)).isoformat()
            one_shot = True
        elif interval_seconds > 0:
            next_run = (datetime.now() + timedelta(seconds=interval_seconds)).isoformat()
        else:
            next_run = self._parse_schedule(schedule)

        job = ScheduledJob(
            id=job_id, name=name, prompt=prompt, model=model, mode=mode,
            cwd=cwd, schedule=schedule, next_run=next_run,
            interval_seconds=interval_seconds, one_shot=one_shot,
        )
        self._jobs[job_id] = job
        self._save_job(job)
        log.info("Scheduled job '%s' (id=%s) next_run=%s", name, job_id, next_run)
        return job

    def remove_job(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._delete_job(job_id)
            return True
        return False

    def list_jobs(self) -> list[dict]:
        return [j.to_dict() for j in self._jobs.values()]

    def get_job(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def toggle_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job:
            job.enabled = not job.enabled
            self._save_job(job)
            return True
        return False

    def _parse_schedule(self, schedule: str) -> str:
        """Parse a simple schedule string into next_run ISO timestamp."""
        lower = schedule.lower().strip()
        now = datetime.now()

        if "every" in lower:
            if "minute" in lower:
                return (now + timedelta(minutes=1)).isoformat()
            elif "hour" in lower:
                return (now + timedelta(hours=1)).isoformat()
            elif "day" in lower:
                return (now + timedelta(days=1)).isoformat()
            elif "week" in lower:
                return (now + timedelta(weeks=1)).isoformat()

        # Default: 1 hour from now
        return (now + timedelta(hours=1)).isoformat()

    def _compute_next_run(self, job: ScheduledJob) -> str:
        """Compute next run time after execution."""
        now = datetime.now()
        if job.interval_seconds > 0:
            return (now + timedelta(seconds=job.interval_seconds)).isoformat()
        return self._parse_schedule(job.schedule)

    async def start(self):
        """Start the scheduler loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        log.info("Scheduler started with %d jobs", len(self._jobs))

    async def stop(self):
        """Stop the scheduler loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        """Main scheduler loop — checks every 10 seconds."""
        while self._running:
            try:
                now = datetime.now()
                for job in list(self._jobs.values()):
                    if not job.enabled or not job.next_run:
                        continue
                    try:
                        next_run = datetime.fromisoformat(job.next_run)
                    except ValueError:
                        continue

                    if now >= next_run:
                        log.info("Executing scheduled job '%s' (id=%s)", job.name, job.id)
                        await self._execute_job(job)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Scheduler loop error: %s", e)

            await asyncio.sleep(10)

    async def _execute_job(self, job: ScheduledJob):
        """Execute a scheduled job."""
        try:
            if self.task_callback:
                await self.task_callback(job.prompt, job.model, job.mode, job.cwd)
                job.last_status = "success"
            else:
                job.last_status = "no_callback"

            job.last_run = datetime.now().isoformat()

            if job.one_shot:
                job.enabled = False
            else:
                job.next_run = self._compute_next_run(job)

            self._save_job(job)

        except Exception as e:
            job.last_status = f"error: {e}"
            job.last_run = datetime.now().isoformat()
            self._save_job(job)
            log.error("Job '%s' failed: %s", job.name, e)
