import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Set

@dataclass
class Task:
    id: str
    fn: Callable[..., Any]
    deps: List[str] = field(default_factory=list)
    retries: int = 2

class TaskEngine:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.cache: Dict[str, Any] = {}
        self.dependents: Dict[str, List[str]] = defaultdict(list)
        self.completed: Set[str] = set()
        self.scheduled: Set[str] = set()
        self._lock = asyncio.Lock()

    def add_task(self, task: Task):
        self.tasks[task.id] = task
        for dep in task.deps:
            self.dependents[dep].append(task.id)

    async def run_task(self, task_id: str):
        if task_id in self.cache:
            return self.cache[task_id]

        task = self.tasks[task_id]

        # run dependencies first
        for dep in task.deps:
            if dep not in self.completed:
                await self.run_task(dep)

        attempt = 0
        while attempt <= task.retries:
            try:
                result = await task.fn()
                self.cache[task_id] = result
                self.completed.add(task_id)
                break
            except Exception as e:
                attempt += 1
                if attempt > task.retries:
                    print(f"Task {task_id} failed after {task.retries} retries: {e}")
                    raise e

        # notify dependents
        for child in self.dependents[task_id]:
            # Check if child task is not completed and not already scheduled
            if child not in self.completed and child not in self.scheduled:
                self.scheduled.add(child)
                asyncio.create_task(self.run_task(child))

    async def run_all(self):
        # Create a list of root tasks (those without dependencies)
        root_tasks = [task_id for task_id, task in self.tasks.items() if not task.deps]
        
        # Schedule all root tasks concurrently
        await asyncio.gather(*[self.run_task(task_id) for task_id in root_tasks])


# ---------------- TEST TASKS ----------------

async def task_a():
    await asyncio.sleep(0.1)
    return "A"

async def task_b():
    await asyncio.sleep(0.1)
    return "B"

async def task_c():
    await asyncio.sleep(0.1)
    return "C"

async def task_d():
    await asyncio.sleep(0.1)
    return "D"


engine = TaskEngine()

engine.add_task(Task("A", task_a))
engine.add_task(Task("B", task_b, deps=["A"]))
engine.add_task(Task("C", task_c, deps=["A"]))
engine.add_task(Task("D", task_d, deps=["B", "C"]))

asyncio.run(engine.run_all())

print(engine.cache)
print(engine.completed)