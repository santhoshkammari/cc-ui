"""
Git service — enhanced git operations for the UI.

Supports:
- Branch management (create, switch, list, delete)
- Status and diff
- Commit with message
- Repo detection
- Log viewing
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

log = logging.getLogger("cc-ui.git")


class GitService:
    @staticmethod
    async def _run(cmd: list[str], cwd: str) -> tuple[int, str, str]:
        """Run a git command and return (returncode, stdout, stderr)."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout.decode().strip(), stderr.decode().strip()

    @staticmethod
    async def is_repo(cwd: str) -> bool:
        rc, _, _ = await GitService._run(
            ["git", "rev-parse", "--is-inside-work-tree"], cwd
        )
        return rc == 0

    @staticmethod
    async def get_status(cwd: str) -> dict:
        """Get comprehensive git status."""
        if not await GitService.is_repo(cwd):
            return {"is_git": False}

        rc, status, _ = await GitService._run(["git", "status", "--short"], cwd)
        rc, branch, _ = await GitService._run(["git", "branch", "--show-current"], cwd)
        rc, diff, _ = await GitService._run(["git", "diff", "HEAD"], cwd)
        rc, remote, _ = await GitService._run(["git", "remote", "-v"], cwd)

        return {
            "is_git": True,
            "branch": branch,
            "status": status,
            "diff": diff,
            "remote": remote.split("\n")[0] if remote else "",
            "has_changes": bool(status.strip()),
        }

    @staticmethod
    async def list_branches(cwd: str) -> list[dict]:
        """List all branches with current marker."""
        if not await GitService.is_repo(cwd):
            return []

        rc, output, _ = await GitService._run(["git", "branch", "-a", "--format=%(refname:short)|%(objectname:short)|%(HEAD)"], cwd)
        branches = []
        for line in output.splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                branches.append({
                    "name": parts[0].strip(),
                    "sha": parts[1].strip(),
                    "current": parts[2].strip() == "*",
                })
        return branches

    @staticmethod
    async def create_branch(cwd: str, name: str, checkout: bool = True) -> dict:
        """Create a new branch, optionally check it out."""
        if not await GitService.is_repo(cwd):
            return {"error": "Not a git repository"}

        if checkout:
            rc, out, err = await GitService._run(["git", "checkout", "-b", name], cwd)
        else:
            rc, out, err = await GitService._run(["git", "branch", name], cwd)

        return {"success": rc == 0, "output": out or err, "branch": name}

    @staticmethod
    async def switch_branch(cwd: str, name: str) -> dict:
        """Switch to an existing branch."""
        rc, out, err = await GitService._run(["git", "checkout", name], cwd)
        return {"success": rc == 0, "output": out or err, "branch": name}

    @staticmethod
    async def commit(cwd: str, message: str, add_all: bool = True) -> dict:
        """Commit changes."""
        if add_all:
            await GitService._run(["git", "add", "-A"], cwd)
        rc, out, err = await GitService._run(["git", "commit", "-m", message], cwd)
        return {"success": rc == 0, "output": out or err}

    @staticmethod
    async def get_log(cwd: str, n: int = 20) -> list[dict]:
        """Get recent commit log."""
        if not await GitService.is_repo(cwd):
            return []

        rc, output, _ = await GitService._run(
            ["git", "log", f"-{n}", "--format=%H|%h|%s|%an|%ai"], cwd
        )
        commits = []
        for line in output.splitlines():
            if not line.strip():
                continue
            parts = line.split("|", 4)
            if len(parts) >= 5:
                commits.append({
                    "sha": parts[0], "short_sha": parts[1],
                    "message": parts[2], "author": parts[3],
                    "date": parts[4],
                })
        return commits

    @staticmethod
    async def get_diff(cwd: str) -> dict:
        """Get status + diff (for the git panel)."""
        if not await GitService.is_repo(cwd):
            return {"is_git": False, "status": "", "diff": ""}

        rc, status, _ = await GitService._run(["git", "status", "--short"], cwd)
        rc, diff, _ = await GitService._run(["git", "diff", "HEAD"], cwd)
        return {"is_git": True, "status": status, "diff": diff}

    @staticmethod
    async def stash(cwd: str, message: str = "") -> dict:
        cmd = ["git", "stash"]
        if message:
            cmd += ["-m", message]
        rc, out, err = await GitService._run(cmd, cwd)
        return {"success": rc == 0, "output": out or err}

    @staticmethod
    async def stash_pop(cwd: str) -> dict:
        rc, out, err = await GitService._run(["git", "stash", "pop"], cwd)
        return {"success": rc == 0, "output": out or err}

    @staticmethod
    async def list_prs(cwd: str, limit: int = 20) -> list[dict]:
        """List open PRs using gh CLI."""
        rc, out, err = await GitService._run(
            ["gh", "pr", "list", "--json",
             "number,title,author,headRefName,baseRefName,state,url,createdAt,isDraft",
             "--limit", str(limit)],
            cwd,
        )
        if rc != 0:
            return []
        import json
        try:
            return json.loads(out)
        except Exception:
            return []

    @staticmethod
    async def create_pr(cwd: str, title: str, body: str = "", base: str = "main", head: str = "") -> dict:
        """Create a PR using gh CLI."""
        cmd = ["gh", "pr", "create", "--title", title, "--body", body, "--base", base]
        if head:
            cmd.extend(["--head", head])
        rc, out, err = await GitService._run(cmd, cwd)
        return {"success": rc == 0, "output": out or err}

    @staticmethod
    async def merge_pr(cwd: str, pr_number: int, method: str = "merge") -> dict:
        """Merge a PR using gh CLI."""
        rc, out, err = await GitService._run(
            ["gh", "pr", "merge", str(pr_number), f"--{method}"], cwd
        )
        return {"success": rc == 0, "output": out or err}

    @staticmethod
    async def pr_diff(cwd: str, pr_number: int) -> str:
        """Get diff for a PR using gh CLI."""
        rc, out, err = await GitService._run(
            ["gh", "pr", "diff", str(pr_number)], cwd
        )
        return out if rc == 0 else err

    @staticmethod
    async def get_all_diffs(cwd: str) -> dict:
        """Get staged, unstaged, and untracked diffs."""
        _, staged, _ = await GitService._run(["git", "diff", "--cached"], cwd)
        _, unstaged, _ = await GitService._run(["git", "diff"], cwd)
        _, untracked_files, _ = await GitService._run(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd
        )
        untracked_diff = ""
        for f in untracked_files.splitlines():
            if f:
                try:
                    fpath = os.path.join(cwd, f)
                    if os.path.isfile(fpath) and os.path.getsize(fpath) < 50000:
                        with open(fpath, 'r', errors='replace') as fh:
                            content = fh.read()
                        untracked_diff += f"\n--- /dev/null\n+++ b/{f}\n" + "\n".join("+" + line for line in content.splitlines()) + "\n"
                except Exception:
                    pass
        return {
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked_diff,
            "total": staged + "\n" + unstaged + "\n" + untracked_diff,
        }
