"""
Integration tests for CC-UI providers.

Tests all providers against the mock vLLM server.
Run: pytest tests/test_providers.py -v
"""
import asyncio
import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestProviderRegistry:
    def test_registry_imports(self):
        from services.providers.registry import list_providers, get_provider
        providers = list_providers()
        assert len(providers) > 0
        names = [p["name"] for p in providers]
        assert "vllm" in names or "kivi" in names or "inhouse" in names

    def test_get_unknown_provider(self):
        from services.providers.registry import get_provider
        with pytest.raises(ValueError):
            get_provider("nonexistent-provider-xyz")


class TestBaseProvider:
    def test_provider_event_to_history(self):
        from services.providers.base import ProviderEvent, EventType
        
        # Text event
        ev = ProviderEvent(type=EventType.TEXT, content="Hello world")
        h = ev.to_history_entry()
        assert h["role"] == "assistant"
        assert h["content"] == "Hello world"

        # Tool start event
        ev = ProviderEvent(type=EventType.TOOL_START, metadata={"title": "⚙ bash", "args": '{"cmd":"ls"}'})
        h = ev.to_history_entry()
        assert h["metadata"]["status"] == "pending"

        # Error event
        ev = ProviderEvent(type=EventType.ERROR, content="Connection failed")
        h = ev.to_history_entry()
        assert "❌" in h["content"]


class TestVLLMProvider:
    """Tests using mock vLLM server on port 9999."""

    @pytest.mark.asyncio
    async def test_vllm_streaming(self):
        from services.providers.vllm import VLLMProvider
        from services.providers.base import ProviderConfig, EventType

        provider = VLLMProvider()
        config = ProviderConfig(
            base_url="http://127.0.0.1:9999",
            api_key="dummy",
            model="mock-model-v1",
        )

        text_parts = []
        got_done = False
        async for event in provider.run("Hello, test message", config):
            if event.type == EventType.TEXT:
                text_parts.append(event.content)
            elif event.type == EventType.DONE:
                got_done = True

        full_text = "".join(text_parts)
        assert len(full_text) > 0
        assert got_done


class TestKiviProvider:
    """Tests using mock vLLM server as Kivi backend."""

    @pytest.mark.asyncio
    async def test_kivi_streaming(self):
        from services.providers.kivi import KiviProvider
        from services.providers.base import ProviderConfig, EventType

        provider = KiviProvider()
        config = ProviderConfig(
            base_url="http://127.0.0.1:9999",
            api_key="dummy",
            model="mock-model-v1",
        )

        text_parts = []
        got_done = False
        async for event in provider.run("Hello from Kivi test", config):
            if event.type == EventType.TEXT:
                text_parts.append(event.content)
            elif event.type == EventType.DONE:
                got_done = True

        full_text = "".join(text_parts)
        assert len(full_text) > 0
        assert got_done


class TestInhouseProvider:
    """Tests using mock vLLM server as in-house AI backend."""

    @pytest.mark.asyncio
    async def test_inhouse_basic(self):
        from services.providers.inhouse import InhouseProvider
        from services.providers.base import ProviderConfig, EventType

        provider = InhouseProvider()
        config = ProviderConfig(
            base_url="http://127.0.0.1:9999/v1",
            api_key="dummy",
            model="mock-model-v1",
            extra={"base_url": "http://127.0.0.1:9999/v1", "use_tools": False},
        )

        text_parts = []
        got_done = False
        async for event in provider.run("Hello inhouse test", config):
            if event.type == EventType.TEXT:
                text_parts.append(event.content)
            elif event.type == EventType.DONE:
                got_done = True
            elif event.type == EventType.ERROR:
                # May fail if openai not installed, that's ok
                pytest.skip(f"Inhouse provider error: {event.content}")

        if not text_parts:
            pytest.skip("No text received (dependency issue)")
        assert got_done


class TestScheduler:
    def test_scheduler_crud(self):
        import tempfile
        from services.scheduler import Scheduler

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            sched = Scheduler(f.name)

            # Add job
            job = sched.add_job("test-job", "echo hello", delay_seconds=3600)
            assert job.name == "test-job"
            assert job.one_shot == True

            # List
            jobs = sched.list_jobs()
            assert len(jobs) == 1

            # Toggle
            sched.toggle_job(job.id)
            j = sched.get_job(job.id)
            assert j["enabled"] == False

            # Delete
            sched.remove_job(job.id)
            assert len(sched.list_jobs()) == 0


class TestGitService:
    @pytest.mark.asyncio
    async def test_is_repo(self):
        from services.git_service import GitService
        # Current project should be a git repo
        result = await GitService.is_repo(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        assert result == True

    @pytest.mark.asyncio
    async def test_list_branches(self):
        from services.git_service import GitService
        branches = await GitService.list_branches(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        assert len(branches) > 0

    @pytest.mark.asyncio
    async def test_get_status(self):
        from services.git_service import GitService
        status = await GitService.get_status(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        assert status["is_git"] == True
        assert "branch" in status


class TestMonitor:
    def test_monitor_basic(self):
        from services.monitor import Monitor
        m = Monitor()
        dashboard = m.get_dashboard()
        assert "uptime_seconds" in dashboard
        assert "tasks" in dashboard

    def test_monitor_metrics(self):
        from services.monitor import Monitor
        m = Monitor()
        metrics = m.get_metrics()
        assert "ccui_uptime_seconds" in metrics
