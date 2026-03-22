# Project 11: Multi-Agent Collaboration Platform (Expert+)

## Goal

Build a **production-grade platform** where multiple specialized AI agents collaborate
autonomously to complete complex, multi-step workflows. Each agent has a distinct role,
tool set, and persona. The platform orchestrates their interactions using sequential,
parallel, and hierarchical execution patterns --- with full observability, human-in-the-loop
controls, and a real-time dashboard showing agent collaboration in action.

The primary use case is an **automated code review pipeline** with five agents:
Planner, Coder, Reviewer, Tester, and PM. A secondary use case demonstrates a
**research team** with Researcher, Analyst, Writer, Editor, and Fact-Checker agents.

---

## Why This Project?

| Question | Answer |
|----------|--------|
| Why multi-agent? | Single-agent systems hit a ceiling on complex tasks. Decomposing work across specialists mirrors how real teams operate and produces higher-quality results. |
| Why not just use LangChain agents? | Off-the-shelf frameworks hide the orchestration logic. Building the orchestration engine yourself teaches distributed systems, state machines, and protocol design. |
| Is this relevant to industry? | Multi-agent architectures are the current frontier --- AutoGen, CrewAI, and internal systems at major tech companies all follow this pattern. |
| What makes this Expert+? | You are building a distributed AI operating system: message passing, state management, tracing, consensus protocols, and a real-time UI --- all at once. |
| Will this stand out in interviews? | Absolutely. This demonstrates systems design, API architecture, observability, and deep understanding of LLM capabilities --- the exact combination senior/staff roles demand. |

---

## Architecture Overview

```
                         ┌───────────────────────────────────┐
                         │     React / Next.js Dashboard     │
                         │  (real-time workflow visualization)│
                         └──────────────┬────────────────────┘
                                        │ WebSocket + REST
                                        ▼
                         ┌───────────────────────────────────┐
                         │         FastAPI Gateway           │
                         │  (workflow CRUD, auth, SSE/WS)    │
                         └──────────────┬────────────────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               ▼                        ▼                        ▼
    ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
    │  Orchestration     │  │  Agent Registry     │  │  Workflow State    │
    │  Engine            │  │  & Factory          │  │  Machine           │
    │  (LangGraph /      │  │  (role definitions, │  │  (PostgreSQL +     │
    │   custom DAG)      │  │   prompt templates) │  │   Redis cache)     │
    └────────┬───────────┘  └────────────────────┘  └────────────────────┘
             │
    ┌────────┴──────────────────────────────────┐
    │           RabbitMQ / Redis Streams         │
    │         (inter-agent message bus)          │
    └──┬──────┬──────┬──────┬──────┬────────────┘
       ▼      ▼      ▼      ▼      ▼
    ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
    │Plan- ││Coder ││Review││Tester││ PM   │
    │ner   ││Agent ││Agent ││Agent ││Agent │
    │Agent ││      ││      ││      ││      │
    └──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘
       │       │       │       │       │
       ▼       ▼       ▼       ▼       ▼
    ┌──────────────────────────────────────────┐
    │           Tool Layer                     │
    │  GitHub API │ Code Exec │ Search │ DB   │
    └──────────────────────────────────────────┘
       │
       ▼
    ┌──────────────────────────────────────────┐
    │        OpenTelemetry Collector           │
    │  Jaeger (traces) │ Prometheus (metrics)  │
    │  Grafana (dashboards)                    │
    └──────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Tool | Why This One |
|----------|------|-------------|
| Language | Python 3.11+ | LLM ecosystem standard, async support |
| LLM Providers | Claude API + OpenAI API | Different models per agent role (e.g., Claude for reasoning, GPT-4o for code) |
| Orchestration | LangGraph / custom DAG engine | Stateful, cyclic agent graphs with conditional routing |
| Message Bus | RabbitMQ (primary) + Redis Streams (lightweight) | Durable inter-agent messaging with acknowledgments |
| State Store | PostgreSQL 16 | Workflow state, agent memory, audit log |
| Cache | Redis 7 | Shared context blackboard, session state, pub/sub |
| API Framework | FastAPI | Async, WebSocket support, auto-generated docs |
| Frontend | React 18 + Next.js 14 | Real-time dashboard with SSR |
| Visualization | React Flow / D3.js | Agent interaction graph rendering |
| Tracing | OpenTelemetry + Jaeger | Distributed tracing across agent calls |
| Metrics | Prometheus + Grafana | Cost tracking, latency, token usage |
| Containerization | Docker + docker-compose | Reproducible multi-service deployment |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Testing | pytest + pytest-asyncio | Async agent testing |
| Code Quality | Ruff + mypy | Fast linting + strict type checking |

---

## Project Structure

```
multi-agent-platform/
│
├── doc/
│   ├── DESIGN_DOC.md                # System design, agent roles, protocols
│   ├── PROJECT_PLAN.md              # This file
│   └── AGENT_SPECS.md               # Per-agent specifications
│
├── pyproject.toml                   # Dependencies and project metadata
├── docker-compose.yaml              # All services orchestrated
│
├── configs/
│   ├── agents/
│   │   ├── planner.yaml             # Planner agent config (model, tools, prompt)
│   │   ├── coder.yaml               # Coder agent config
│   │   ├── reviewer.yaml            # Reviewer agent config
│   │   ├── tester.yaml              # Tester agent config
│   │   └── pm.yaml                  # PM agent config
│   ├── workflows/
│   │   ├── code_review.yaml         # Code review pipeline definition
│   │   └── research_team.yaml       # Research team pipeline definition
│   ├── orchestration.yaml           # Routing rules, timeouts, retries
│   └── observability.yaml           # Tracing, metrics, logging config
│
├── src/
│   ├── __init__.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseAgent abstract class
│   │   ├── registry.py              # Agent registry and factory
│   │   ├── memory.py                # Agent memory (short-term + long-term)
│   │   ├── planner.py               # Planner agent implementation
│   │   ├── coder.py                 # Coder agent implementation
│   │   ├── reviewer.py              # Reviewer agent implementation
│   │   ├── tester.py                # Tester agent implementation
│   │   └── pm.py                    # PM agent implementation
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── engine.py                # Core orchestration engine
│   │   ├── patterns.py              # Sequential, parallel, hierarchical patterns
│   │   ├── router.py                # Dynamic task routing logic
│   │   ├── state_machine.py         # Workflow state management
│   │   └── decomposer.py           # Task decomposition logic
│   │
│   ├── communication/
│   │   ├── __init__.py
│   │   ├── message.py               # Message schema and serialization
│   │   ├── bus.py                   # Message bus abstraction (RabbitMQ/Redis)
│   │   ├── blackboard.py            # Shared context store
│   │   ├── debate.py                # Debate and consensus mechanisms
│   │   └── protocols.py             # Communication protocol definitions
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseTool interface
│   │   ├── github.py                # GitHub API tool
│   │   ├── code_executor.py         # Sandboxed code execution
│   │   ├── search.py                # Web/doc search tool
│   │   └── file_manager.py          # File read/write tool
│   │
│   ├── hitl/
│   │   ├── __init__.py
│   │   ├── escalation.py            # Escalation trigger logic
│   │   ├── approval.py              # Approval gate implementation
│   │   ├── override.py              # Human override mechanisms
│   │   └── feedback.py              # Feedback collection and routing
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── task_completion.py       # End-to-end task scoring
│   │   ├── agent_scoring.py         # Per-agent contribution metrics
│   │   ├── conversation_quality.py  # Inter-agent conversation analysis
│   │   └── cost_analyzer.py         # Token usage and cost tracking
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI application
│   │   ├── routes/
│   │   │   ├── workflows.py         # Workflow CRUD endpoints
│   │   │   ├── agents.py            # Agent management endpoints
│   │   │   └── websocket.py         # Real-time event streaming
│   │   ├── schemas.py               # Pydantic request/response models
│   │   ├── middleware.py            # Auth, rate limiting, CORS
│   │   └── dependencies.py          # Dependency injection
│   │
│   ├── dashboard/
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── components/
│   │   │   │   ├── AgentGraph.tsx    # Real-time agent interaction graph
│   │   │   │   ├── WorkflowTimeline.tsx
│   │   │   │   ├── MessageLog.tsx
│   │   │   │   └── CostTracker.tsx
│   │   │   └── pages/
│   │   │       ├── index.tsx         # Dashboard home
│   │   │       └── workflow/[id].tsx # Individual workflow view
│   │   └── next.config.js
│   │
│   └── monitoring/
│       ├── __init__.py
│       ├── tracing.py               # OpenTelemetry setup
│       ├── metrics.py               # Prometheus metric definitions
│       └── interaction_graph.py     # Agent interaction analysis
│
├── tests/
│   ├── unit/
│   │   ├── test_base_agent.py
│   │   ├── test_message_schema.py
│   │   ├── test_state_machine.py
│   │   ├── test_router.py
│   │   └── test_decomposer.py
│   ├── integration/
│   │   ├── test_agent_communication.py
│   │   ├── test_workflow_execution.py
│   │   └── test_hitl_flow.py
│   ├── e2e/
│   │   ├── test_code_review_pipeline.py
│   │   └── test_research_team.py
│   └── conftest.py
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.agent-worker
│   ├── Dockerfile.dashboard
│   └── Dockerfile.tracing
│
├── infrastructure/
│   ├── rabbitmq/
│   │   └── rabbitmq.conf
│   ├── prometheus/
│   │   └── prometheus.yml
│   ├── grafana/
│   │   └── dashboards/
│   │       ├── agent_overview.json
│   │       └── workflow_costs.json
│   └── jaeger/
│       └── jaeger-config.yaml
│
├── scripts/
│   ├── setup.sh
│   ├── run_workflow.sh
│   └── seed_agents.py
│
└── .github/
    └── workflows/
        ├── ci.yaml
        └── cd.yaml
```

---

## Phase 1: Setup & Design Doc

**Duration:** 2--3 days
**Objective:** Define the system architecture, agent roles, communication protocols, and
workflow patterns before writing any orchestration code.

### Task 1.1: Initialize Repository

Create the project skeleton, set up dependency management, and configure dev tooling.

```bash
mkdir multi-agent-platform && cd multi-agent-platform
git init
python -m venv .venv && source .venv/bin/activate
```

**pyproject.toml core dependencies:**

```toml
[project]
name = "multi-agent-platform"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "openai>=1.50.0",
    "langgraph>=0.2.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.9.0",
    "sqlalchemy>=2.0.35",
    "asyncpg>=0.30.0",
    "redis>=5.2.0",
    "aio-pika>=9.4.0",         # async RabbitMQ
    "opentelemetry-api>=1.28.0",
    "opentelemetry-sdk>=1.28.0",
    "opentelemetry-exporter-otlp>=1.28.0",
    "prometheus-client>=0.21.0",
    "httpx>=0.27.0",
    "structlog>=24.4.0",
    "pyyaml>=6.0.2",
]
```

### Task 1.2: Write the Design Document

The design doc is the most important artifact in a multi-agent system. It must answer:

1. **Agent Role Definitions** --- What does each agent do? What are its boundaries?
2. **Communication Protocol** --- How do agents talk to each other? What is the message
   schema? Is it request/reply, publish/subscribe, or blackboard?
3. **Workflow Patterns** --- When do agents run sequentially vs. in parallel? Who decides?
4. **Failure Modes** --- What happens when an agent hallucinates, loops, or times out?
5. **Human Escalation** --- When does a human need to intervene? How is that triggered?

### Task 1.3: Define Agent Role Specifications

Create a YAML specification for each agent. This becomes the single source of truth for
what each agent can do.

```yaml
# configs/agents/reviewer.yaml
agent:
  name: "Reviewer"
  role: "code_reviewer"
  description: >
    Reviews code changes for correctness, style, security vulnerabilities,
    and adherence to project conventions. Produces structured feedback with
    severity levels (critical, warning, suggestion).

  model:
    provider: "anthropic"
    model_id: "claude-sonnet-4-20250514"
    max_tokens: 4096
    temperature: 0.2

  system_prompt_template: "prompts/reviewer_system.txt"

  tools:
    - github_read_file
    - github_list_changes
    - search_codebase
    - read_documentation

  input_schema:
    required: ["pull_request_url", "review_focus"]
    optional: ["previous_reviews", "style_guide"]

  output_schema:
    type: "review_report"
    fields:
      - name: "findings"
        type: "list[Finding]"
      - name: "overall_verdict"
        type: "enum[approve, request_changes, comment]"
      - name: "confidence"
        type: "float"

  constraints:
    max_retries: 2
    timeout_seconds: 120
    max_tool_calls: 15
    escalate_on: ["confidence < 0.5", "security_finding.critical"]
```

### Task 1.4: Define Communication Protocol

Document the message format that all agents use. Every inter-agent message must follow
this schema so the orchestrator can inspect, route, and log them uniformly.

```python
# src/communication/message.py
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import uuid


class MessageType(str, Enum):
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    CLARIFICATION_REQUEST = "clarification_request"
    CLARIFICATION_RESPONSE = "clarification_response"
    FEEDBACK = "feedback"
    ESCALATION = "escalation"
    STATUS_UPDATE = "status_update"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentMessage(BaseModel):
    """Standard message format for all inter-agent communication."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    parent_message_id: str | None = None

    sender: str            # agent role name
    recipient: str         # agent role name or "orchestrator"
    message_type: MessageType
    priority: Priority = Priority.MEDIUM

    subject: str           # brief description of the message purpose
    content: dict[str, Any]  # structured payload (varies by message type)

    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: str | None = None  # OpenTelemetry trace ID
    span_id: str | None = None   # OpenTelemetry span ID
```

### Task 1.5: Design Workflow Patterns

Define the three core orchestration patterns the engine must support:

| Pattern | Description | Example |
|---------|-------------|---------|
| **Sequential** | Agent A completes, output feeds to Agent B, then C | Planner -> Coder -> Reviewer |
| **Parallel Fan-Out** | Multiple agents work simultaneously on sub-tasks | Reviewer + Tester run concurrently on Coder output |
| **Hierarchical** | A supervisor agent delegates to sub-agents and aggregates | PM supervises Planner, who delegates to Coder + Reviewer |

**Deliverables:**
- [ ] DESIGN_DOC.md with all sections above
- [ ] AGENT_SPECS.md with YAML definitions for all 5 agents
- [ ] Message schema module with Pydantic models
- [ ] Workflow pattern diagrams (ASCII or draw.io)

---

## Phase 2: Agent Framework

**Duration:** 4--5 days
**Objective:** Build the base agent class, prompt management system, tool assignment, and
agent registry so that new agents can be created from configuration alone.

### Task 2.1: Implement the Base Agent Class

The base agent is the foundation of the entire system. It must handle LLM calls, tool
execution, memory management, and structured output --- all behind a clean async interface.

```python
# src/agents/base.py
from abc import ABC, abstractmethod
from typing import Any
import structlog

from src.communication.message import AgentMessage, MessageType
from src.agents.memory import AgentMemory
from src.monitoring.tracing import trace_agent_call


logger = structlog.get_logger()


class BaseAgent(ABC):
    """Abstract base class for all specialized agents."""

    def __init__(
        self,
        name: str,
        role: str,
        model_provider: str,
        model_id: str,
        system_prompt: str,
        tools: list[str],
        constraints: dict[str, Any],
    ):
        self.name = name
        self.role = role
        self.model_provider = model_provider
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.tools = tools
        self.constraints = constraints
        self.memory = AgentMemory(agent_name=name)
        self._call_count = 0

    @trace_agent_call
    async def process(self, message: AgentMessage) -> AgentMessage:
        """Main entry point: receive a message, think, act, respond."""
        logger.info(
            "agent.processing",
            agent=self.name,
            message_type=message.message_type,
            workflow_id=message.workflow_id,
        )

        # 1. Update memory with incoming message
        self.memory.add_message(message)

        # 2. Build the conversation for the LLM
        conversation = self._build_conversation(message)

        # 3. Execute the agent loop (LLM call + tool use)
        result = await self._execute(conversation)

        # 4. Validate output against schema
        validated = self._validate_output(result)

        # 5. Check escalation triggers
        if self._should_escalate(validated):
            return self._create_escalation(message, validated)

        # 6. Build response message
        return AgentMessage(
            workflow_id=message.workflow_id,
            parent_message_id=message.message_id,
            sender=self.role,
            recipient="orchestrator",
            message_type=MessageType.TASK_RESULT,
            subject=f"{self.name} completed task",
            content=validated,
        )

    @abstractmethod
    async def _execute(self, conversation: list[dict]) -> dict[str, Any]:
        """Subclasses implement the actual LLM interaction loop."""
        ...

    @abstractmethod
    def _build_conversation(self, message: AgentMessage) -> list[dict]:
        """Build the LLM conversation from memory and current message."""
        ...

    def _should_escalate(self, result: dict[str, Any]) -> bool:
        """Check if the result triggers any escalation rules."""
        for rule in self.constraints.get("escalate_on", []):
            if self._evaluate_rule(rule, result):
                logger.warning("agent.escalation_triggered", agent=self.name, rule=rule)
                return True
        return False

    def _validate_output(self, result: dict[str, Any]) -> dict[str, Any]:
        """Validate agent output against the configured output schema."""
        # Pydantic validation based on agent config
        return result

    def _evaluate_rule(self, rule: str, result: dict[str, Any]) -> bool:
        """Evaluate a simple escalation rule expression."""
        # Parse rules like "confidence < 0.5" or "security_finding.critical"
        ...
```

### Task 2.2: System Prompt Templates

Create a prompt template system with Jinja2-style rendering. Each agent's system prompt
is stored as a text file with placeholders for dynamic context.

```
# prompts/reviewer_system.txt
You are the **Reviewer Agent** on a collaborative software engineering team.

## Your Role
You review code changes for:
- Correctness: Does the code do what it claims?
- Security: Are there vulnerabilities (injection, auth bypass, data leaks)?
- Style: Does it follow the project's coding conventions?
- Performance: Are there obvious inefficiencies?

## Your Constraints
- You MUST produce structured output with severity levels.
- You MUST NOT rewrite the code yourself --- suggest changes only.
- If you are less than 50% confident in a finding, flag it as "needs_human_review".

## Current Context
Project: {{ project_name }}
Language: {{ language }}
Style Guide: {{ style_guide_summary }}
Previous Reviews: {{ previous_review_count }} reviews in this PR

## Output Format
Respond with a JSON object:
{
  "findings": [
    {
      "file": "path/to/file.py",
      "line_range": [10, 15],
      "severity": "critical|warning|suggestion",
      "category": "correctness|security|style|performance",
      "description": "...",
      "suggested_fix": "...",
      "confidence": 0.0-1.0
    }
  ],
  "overall_verdict": "approve|request_changes|comment",
  "summary": "Brief overall assessment",
  "confidence": 0.0-1.0
}
```

### Task 2.3: Tool Assignment and Execution

Each agent gets a specific set of tools. The tool layer provides a uniform interface so
agents do not need to know the implementation details of each tool.

```python
# src/tools/base.py
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    data: Any
    error: str | None = None
    execution_time_ms: float


class BaseTool(ABC):
    """Interface that all tools must implement."""

    name: str
    description: str
    parameters_schema: dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters."""
        ...

    def to_llm_schema(self) -> dict[str, Any]:
        """Convert to the function-calling schema expected by the LLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
```

### Task 2.4: Agent Registry and Factory

The registry allows the orchestration engine to look up agents by role and instantiate
them from YAML configuration.

```python
# src/agents/registry.py
from typing import Type
import yaml

from src.agents.base import BaseAgent


class AgentRegistry:
    """Central registry for agent types and configurations."""

    _agent_types: dict[str, Type[BaseAgent]] = {}
    _instances: dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, role: str):
        """Decorator to register an agent class for a given role."""
        def wrapper(agent_class: Type[BaseAgent]):
            cls._agent_types[role] = agent_class
            return agent_class
        return wrapper

    @classmethod
    def create_from_config(cls, config_path: str) -> BaseAgent:
        """Instantiate an agent from a YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        agent_config = config["agent"]
        role = agent_config["role"]

        if role not in cls._agent_types:
            raise ValueError(f"No agent class registered for role: {role}")

        agent_class = cls._agent_types[role]
        system_prompt = cls._load_prompt_template(
            agent_config["system_prompt_template"]
        )

        agent = agent_class(
            name=agent_config["name"],
            role=role,
            model_provider=agent_config["model"]["provider"],
            model_id=agent_config["model"]["model_id"],
            system_prompt=system_prompt,
            tools=agent_config["tools"],
            constraints=agent_config.get("constraints", {}),
        )

        cls._instances[role] = agent
        return agent

    @classmethod
    def get(cls, role: str) -> BaseAgent:
        """Retrieve a registered agent instance by role."""
        if role not in cls._instances:
            raise KeyError(f"Agent not instantiated for role: {role}")
        return cls._instances[role]
```

### Task 2.5: Agent Memory System

Each agent needs short-term memory (current conversation) and long-term memory
(lessons learned, past decisions) to maintain coherence across workflow steps.

```python
# src/agents/memory.py
from collections import deque
from datetime import datetime
from typing import Any

from src.communication.message import AgentMessage


class AgentMemory:
    """Manages short-term and long-term memory for an agent."""

    def __init__(self, agent_name: str, short_term_limit: int = 50):
        self.agent_name = agent_name
        self.short_term: deque[AgentMessage] = deque(maxlen=short_term_limit)
        self.long_term: list[dict[str, Any]] = []
        self.context_summary: str = ""

    def add_message(self, message: AgentMessage) -> None:
        """Add a message to short-term memory."""
        self.short_term.append(message)

    def add_lesson(self, lesson: str, tags: list[str] | None = None) -> None:
        """Store a lesson learned in long-term memory."""
        self.long_term.append({
            "lesson": lesson,
            "tags": tags or [],
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_relevant_context(self, query: str, max_items: int = 5) -> list[str]:
        """Retrieve relevant long-term memories for the current task."""
        # Simple keyword matching; upgrade to embedding similarity later
        relevant = []
        query_words = set(query.lower().split())
        for item in self.long_term:
            tag_overlap = query_words & set(t.lower() for t in item["tags"])
            if tag_overlap:
                relevant.append(item["lesson"])
        return relevant[:max_items]

    def summarize_conversation(self) -> str:
        """Create a compressed summary of the current conversation."""
        messages = list(self.short_term)
        if len(messages) <= 5:
            return "\n".join(
                f"[{m.sender}] {m.subject}" for m in messages
            )
        # For longer conversations, keep first 2, last 3, summarize middle
        summary_parts = [f"[{m.sender}] {m.subject}" for m in messages[:2]]
        summary_parts.append(f"... ({len(messages) - 5} messages omitted) ...")
        summary_parts.extend(f"[{m.sender}] {m.subject}" for m in messages[-3:])
        return "\n".join(summary_parts)
```

**Deliverables:**
- [ ] `BaseAgent` abstract class with process/execute/validate/escalate lifecycle
- [ ] Prompt template system with Jinja2 rendering
- [ ] `BaseTool` interface and at least 2 concrete tools (GitHub, code executor)
- [ ] `AgentRegistry` with decorator-based registration and YAML factory
- [ ] `AgentMemory` with short-term and long-term storage
- [ ] Unit tests for all components

---

## Phase 3: Orchestration Engine

**Duration:** 5--7 days
**Objective:** Build the brain of the platform --- the engine that decomposes tasks, routes
them to agents, manages execution patterns, and tracks workflow state.

### Task 3.1: Workflow State Machine

The state machine tracks every workflow through its lifecycle. It persists state to
PostgreSQL so workflows survive restarts.

```python
# src/orchestration/state_machine.py
from enum import Enum
from typing import Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStep(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_role: str
    task_description: str
    dependencies: list[str] = []       # step_ids that must complete first
    status: StepStatus = StepStatus.PENDING
    result: dict[str, Any] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_count: int = 0
    max_retries: int = 2


class WorkflowState(BaseModel):
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    steps: list[WorkflowStep]
    shared_context: dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_ready_steps(self) -> list[WorkflowStep]:
        """Return steps whose dependencies are all completed."""
        completed_ids = {
            s.step_id for s in self.steps if s.status == StepStatus.COMPLETED
        }
        return [
            s for s in self.steps
            if s.status == StepStatus.PENDING
            and all(dep in completed_ids for dep in s.dependencies)
        ]

    def is_complete(self) -> bool:
        """Check if all steps are in a terminal state."""
        terminal = {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}
        return all(s.status in terminal for s in self.steps)
```

### Task 3.2: Task Decomposition

The decomposer takes a high-level task description and breaks it into concrete steps
with dependency relationships. It can use an LLM (the Planner agent) or rule-based
decomposition from workflow templates.

```python
# src/orchestration/decomposer.py
from typing import Any

from src.orchestration.state_machine import WorkflowStep


class TaskDecomposer:
    """Breaks high-level tasks into agent-assignable steps."""

    def __init__(self, workflow_templates: dict[str, Any]):
        self.templates = workflow_templates

    def decompose_from_template(
        self, template_name: str, context: dict[str, Any]
    ) -> list[WorkflowStep]:
        """Use a predefined template to create workflow steps."""
        template = self.templates[template_name]
        steps = []
        step_id_map: dict[str, str] = {}

        for step_def in template["steps"]:
            step = WorkflowStep(
                agent_role=step_def["agent"],
                task_description=step_def["task"].format(**context),
                dependencies=[
                    step_id_map[dep] for dep in step_def.get("depends_on", [])
                ],
                max_retries=step_def.get("max_retries", 2),
            )
            step_id_map[step_def["name"]] = step.step_id
            steps.append(step)

        return steps

    async def decompose_with_planner(
        self, task: str, planner_agent: Any
    ) -> list[WorkflowStep]:
        """Use the Planner agent to dynamically decompose a task."""
        # Send the task to the Planner agent for decomposition
        # Parse the structured output into WorkflowSteps
        ...
```

### Task 3.3: Execution Patterns

Implement the three core patterns: sequential, parallel fan-out, and hierarchical
delegation.

```python
# src/orchestration/patterns.py
import asyncio
from typing import Any

from src.agents.registry import AgentRegistry
from src.communication.message import AgentMessage, MessageType
from src.orchestration.state_machine import WorkflowState, WorkflowStep, StepStatus


class SequentialPattern:
    """Execute steps one after another, passing output as input."""

    async def execute(
        self, steps: list[WorkflowStep], state: WorkflowState
    ) -> list[dict[str, Any]]:
        results = []
        for step in steps:
            agent = AgentRegistry.get(step.agent_role)
            message = AgentMessage(
                workflow_id=state.workflow_id,
                sender="orchestrator",
                recipient=step.agent_role,
                message_type=MessageType.TASK_ASSIGNMENT,
                subject=step.task_description,
                content={
                    "task": step.task_description,
                    "previous_results": results,
                    "shared_context": state.shared_context,
                },
            )
            result = await agent.process(message)
            step.status = StepStatus.COMPLETED
            results.append(result.content)
        return results


class ParallelFanOutPattern:
    """Execute independent steps concurrently, then collect results."""

    async def execute(
        self, steps: list[WorkflowStep], state: WorkflowState
    ) -> list[dict[str, Any]]:
        tasks = []
        for step in steps:
            agent = AgentRegistry.get(step.agent_role)
            message = AgentMessage(
                workflow_id=state.workflow_id,
                sender="orchestrator",
                recipient=step.agent_role,
                message_type=MessageType.TASK_ASSIGNMENT,
                subject=step.task_description,
                content={
                    "task": step.task_description,
                    "shared_context": state.shared_context,
                },
            )
            tasks.append(agent.process(message))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for step, result in zip(steps, results):
            if isinstance(result, Exception):
                step.status = StepStatus.FAILED
                processed.append({"error": str(result)})
            else:
                step.status = StepStatus.COMPLETED
                processed.append(result.content)
        return processed


class HierarchicalPattern:
    """A supervisor agent delegates to sub-agents and aggregates."""

    async def execute(
        self,
        supervisor_role: str,
        sub_steps: list[WorkflowStep],
        state: WorkflowState,
    ) -> dict[str, Any]:
        supervisor = AgentRegistry.get(supervisor_role)

        # Supervisor decides how to delegate
        plan_message = AgentMessage(
            workflow_id=state.workflow_id,
            sender="orchestrator",
            recipient=supervisor_role,
            message_type=MessageType.TASK_ASSIGNMENT,
            subject="Plan delegation for sub-tasks",
            content={
                "sub_tasks": [s.task_description for s in sub_steps],
                "available_agents": [s.agent_role for s in sub_steps],
            },
        )
        delegation_plan = await supervisor.process(plan_message)

        # Execute delegated work (could be sequential or parallel)
        fan_out = ParallelFanOutPattern()
        results = await fan_out.execute(sub_steps, state)

        # Supervisor aggregates results
        aggregate_message = AgentMessage(
            workflow_id=state.workflow_id,
            sender="orchestrator",
            recipient=supervisor_role,
            message_type=MessageType.TASK_ASSIGNMENT,
            subject="Aggregate sub-agent results",
            content={"results": results},
        )
        return (await supervisor.process(aggregate_message)).content
```

### Task 3.4: Dynamic Routing Logic

The router inspects step results and decides what happens next. It handles conditional
branching (e.g., if the Reviewer requests changes, route back to the Coder).

```python
# src/orchestration/router.py
from typing import Any

from src.orchestration.state_machine import WorkflowState, WorkflowStep


class WorkflowRouter:
    """Decides the next step(s) based on current state and results."""

    def __init__(self, routing_rules: dict[str, Any]):
        self.rules = routing_rules

    def get_next_steps(
        self, state: WorkflowState, latest_result: dict[str, Any]
    ) -> list[WorkflowStep]:
        """Evaluate routing rules and return the next steps to execute."""
        ready = state.get_ready_steps()

        # Apply conditional routing rules
        for rule in self.rules.get("conditions", []):
            if self._evaluate_condition(rule["if"], latest_result):
                return self._apply_action(rule["then"], state)

        return ready

    def _evaluate_condition(
        self, condition: dict[str, Any], result: dict[str, Any]
    ) -> bool:
        """Evaluate a condition against a step result."""
        field = condition["field"]
        operator = condition["operator"]
        value = condition["value"]

        actual = result.get(field)
        if operator == "equals":
            return actual == value
        elif operator == "less_than":
            return actual < value
        elif operator == "contains":
            return value in actual
        return False
```

### Task 3.5: Core Orchestration Engine

Tie everything together into the main engine that drives workflow execution end to end.

```python
# src/orchestration/engine.py
import asyncio
import structlog

from src.orchestration.state_machine import WorkflowState, WorkflowStatus, StepStatus
from src.orchestration.decomposer import TaskDecomposer
from src.orchestration.router import WorkflowRouter
from src.orchestration.patterns import (
    SequentialPattern,
    ParallelFanOutPattern,
)

logger = structlog.get_logger()


class OrchestrationEngine:
    """Main engine that drives multi-agent workflow execution."""

    def __init__(
        self,
        decomposer: TaskDecomposer,
        router: WorkflowRouter,
        state_store: Any,  # PostgreSQL-backed store
    ):
        self.decomposer = decomposer
        self.router = router
        self.state_store = state_store
        self.sequential = SequentialPattern()
        self.parallel = ParallelFanOutPattern()

    async def run_workflow(self, state: WorkflowState) -> WorkflowState:
        """Execute a workflow to completion."""
        state.status = WorkflowStatus.RUNNING
        await self.state_store.save(state)

        while not state.is_complete():
            ready_steps = state.get_ready_steps()
            if not ready_steps:
                logger.error("workflow.deadlock", workflow_id=state.workflow_id)
                state.status = WorkflowStatus.FAILED
                break

            logger.info(
                "workflow.executing_steps",
                workflow_id=state.workflow_id,
                step_count=len(ready_steps),
                agents=[s.agent_role for s in ready_steps],
            )

            if len(ready_steps) == 1:
                results = await self.sequential.execute(ready_steps, state)
            else:
                results = await self.parallel.execute(ready_steps, state)

            # Update shared context with results
            for step, result in zip(ready_steps, results):
                state.shared_context[step.step_id] = result

            # Route to next steps (handles conditional branching)
            if results:
                next_steps = self.router.get_next_steps(state, results[-1])
                # Dynamic steps get appended if routing creates new work
                for ns in next_steps:
                    if ns.step_id not in {s.step_id for s in state.steps}:
                        state.steps.append(ns)

            await self.state_store.save(state)

        if state.is_complete():
            state.status = WorkflowStatus.COMPLETED
        await self.state_store.save(state)
        return state
```

**Deliverables:**
- [ ] `WorkflowState` and `WorkflowStep` models with PostgreSQL persistence
- [ ] `TaskDecomposer` with template-based and LLM-based decomposition
- [ ] Sequential, parallel, and hierarchical execution patterns
- [ ] `WorkflowRouter` with conditional branching rules
- [ ] `OrchestrationEngine` that drives end-to-end execution
- [ ] Integration tests: run a 3-step sequential workflow with mock agents

---

## Phase 4: Communication Layer

**Duration:** 4--5 days
**Objective:** Build the inter-agent communication infrastructure: message bus integration,
shared context blackboard, and debate/consensus mechanisms.

### Task 4.1: Message Bus Abstraction

Create an abstraction over RabbitMQ and Redis Streams so the system can switch backends
without changing agent code.

```python
# src/communication/bus.py
from abc import ABC, abstractmethod
from typing import Callable, Awaitable
import json
import aio_pika
import redis.asyncio as redis

from src.communication.message import AgentMessage


class MessageBus(ABC):
    """Abstract message bus for inter-agent communication."""

    @abstractmethod
    async def publish(self, channel: str, message: AgentMessage) -> None: ...

    @abstractmethod
    async def subscribe(
        self, channel: str, handler: Callable[[AgentMessage], Awaitable[None]]
    ) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class RabbitMQBus(MessageBus):
    """RabbitMQ implementation with durable queues and acknowledgments."""

    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self._connection: aio_pika.Connection | None = None
        self._channel: aio_pika.Channel | None = None

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(self.connection_url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=1)

    async def publish(self, channel: str, message: AgentMessage) -> None:
        if not self._channel:
            raise RuntimeError("Not connected to RabbitMQ")

        exchange = await self._channel.declare_exchange(
            "agent_messages", aio_pika.ExchangeType.TOPIC, durable=True
        )
        await exchange.publish(
            aio_pika.Message(
                body=message.model_dump_json().encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers={"sender": message.sender, "type": message.message_type},
            ),
            routing_key=channel,
        )

    async def subscribe(
        self, channel: str, handler: Callable[[AgentMessage], Awaitable[None]]
    ) -> None:
        if not self._channel:
            raise RuntimeError("Not connected to RabbitMQ")

        queue = await self._channel.declare_queue(channel, durable=True)
        exchange = await self._channel.declare_exchange(
            "agent_messages", aio_pika.ExchangeType.TOPIC, durable=True
        )
        await queue.bind(exchange, routing_key=channel)

        async def on_message(msg: aio_pika.IncomingMessage):
            async with msg.process():
                agent_msg = AgentMessage.model_validate_json(msg.body)
                await handler(agent_msg)

        await queue.consume(on_message)

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()


class RedisBus(MessageBus):
    """Redis Streams implementation for lighter-weight messaging."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: redis.Redis | None = None

    async def connect(self) -> None:
        self._redis = redis.from_url(self.redis_url)

    async def publish(self, channel: str, message: AgentMessage) -> None:
        if not self._redis:
            raise RuntimeError("Not connected to Redis")
        await self._redis.xadd(
            channel,
            {"data": message.model_dump_json()},
            maxlen=10000,
        )

    async def subscribe(
        self, channel: str, handler: Callable[[AgentMessage], Awaitable[None]]
    ) -> None:
        if not self._redis:
            raise RuntimeError("Not connected to Redis")
        last_id = "0"
        while True:
            entries = await self._redis.xread(
                {channel: last_id}, count=10, block=5000
            )
            for stream, messages in entries:
                for msg_id, data in messages:
                    agent_msg = AgentMessage.model_validate_json(data[b"data"])
                    await handler(agent_msg)
                    last_id = msg_id

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
```

### Task 4.2: Shared Context Blackboard

The blackboard is a shared memory space where agents post findings that other agents
can read. Unlike direct messages, blackboard entries are visible to all agents in the
workflow.

```python
# src/communication/blackboard.py
from datetime import datetime
from typing import Any
import redis.asyncio as redis
import json


class Blackboard:
    """Shared context store accessible to all agents in a workflow."""

    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client

    async def post(
        self,
        workflow_id: str,
        agent_role: str,
        key: str,
        value: Any,
        tags: list[str] | None = None,
    ) -> None:
        """Post a finding or context item to the blackboard."""
        entry = {
            "agent": agent_role,
            "key": key,
            "value": value,
            "tags": tags or [],
            "timestamp": datetime.utcnow().isoformat(),
        }
        bb_key = f"blackboard:{workflow_id}"
        await self._redis.hset(bb_key, key, json.dumps(entry))
        await self._redis.expire(bb_key, 86400)  # 24h TTL

    async def read(self, workflow_id: str, key: str) -> Any | None:
        """Read a specific entry from the blackboard."""
        raw = await self._redis.hget(f"blackboard:{workflow_id}", key)
        if raw:
            return json.loads(raw)["value"]
        return None

    async def read_all(self, workflow_id: str) -> dict[str, Any]:
        """Read all blackboard entries for a workflow."""
        raw = await self._redis.hgetall(f"blackboard:{workflow_id}")
        return {
            k.decode(): json.loads(v)
            for k, v in raw.items()
        }

    async def read_by_agent(
        self, workflow_id: str, agent_role: str
    ) -> dict[str, Any]:
        """Read all entries posted by a specific agent."""
        all_entries = await self.read_all(workflow_id)
        return {
            k: v for k, v in all_entries.items()
            if v.get("agent") == agent_role
        }
```

### Task 4.3: Debate and Consensus Mechanism

When agents disagree (e.g., Reviewer says the code is insecure but Coder disagrees),
the system needs a structured way to resolve conflicts.

```python
# src/communication/debate.py
from typing import Any
from enum import Enum

from src.agents.base import BaseAgent
from src.communication.message import AgentMessage, MessageType


class ConsensusStrategy(str, Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_CONFIDENCE = "weighted_confidence"
    SUPERVISOR_DECIDES = "supervisor_decides"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class DebateManager:
    """Manages structured debates between agents when they disagree."""

    def __init__(self, max_rounds: int = 3, strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_CONFIDENCE):
        self.max_rounds = max_rounds
        self.strategy = strategy

    async def resolve(
        self,
        topic: str,
        positions: dict[str, dict[str, Any]],  # agent_role -> position
        agents: dict[str, BaseAgent],
        workflow_id: str,
    ) -> dict[str, Any]:
        """Run a structured debate and reach consensus."""
        debate_log: list[dict[str, Any]] = []

        for round_num in range(self.max_rounds):
            new_positions = {}
            for role, agent in agents.items():
                # Each agent sees other agents' positions and can revise
                message = AgentMessage(
                    workflow_id=workflow_id,
                    sender="debate_manager",
                    recipient=role,
                    message_type=MessageType.CLARIFICATION_REQUEST,
                    subject=f"Debate round {round_num + 1}: {topic}",
                    content={
                        "topic": topic,
                        "your_position": positions[role],
                        "other_positions": {
                            r: p for r, p in positions.items() if r != role
                        },
                        "round": round_num + 1,
                        "instruction": "Review other positions. Revise yours if "
                                       "persuaded, or strengthen your argument.",
                    },
                )
                response = await agent.process(message)
                new_positions[role] = response.content

            debate_log.append({"round": round_num + 1, "positions": new_positions})
            positions = new_positions

            # Check if consensus was reached
            if self._check_consensus(positions):
                return {
                    "consensus_reached": True,
                    "final_position": self._aggregate(positions),
                    "rounds": round_num + 1,
                    "debate_log": debate_log,
                }

        # No consensus after max rounds --- apply fallback strategy
        return {
            "consensus_reached": False,
            "resolution_strategy": self.strategy,
            "final_position": self._apply_strategy(positions),
            "rounds": self.max_rounds,
            "debate_log": debate_log,
        }

    def _check_consensus(self, positions: dict[str, dict[str, Any]]) -> bool:
        """Check if all agents agree on the core verdict."""
        verdicts = [p.get("verdict") for p in positions.values()]
        return len(set(verdicts)) == 1

    def _aggregate(self, positions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Aggregate positions when consensus is reached."""
        # Merge all findings, deduplicate
        return list(positions.values())[0]

    def _apply_strategy(
        self, positions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Apply fallback consensus strategy."""
        if self.strategy == ConsensusStrategy.WEIGHTED_CONFIDENCE:
            best_role = max(
                positions,
                key=lambda r: positions[r].get("confidence", 0),
            )
            return positions[best_role]
        elif self.strategy == ConsensusStrategy.ESCALATE_TO_HUMAN:
            return {
                "needs_human_decision": True,
                "positions": positions,
            }
        return list(positions.values())[0]
```

### Task 4.4: RabbitMQ Infrastructure Setup

```yaml
# docker-compose.yaml (relevant services)
services:
  rabbitmq:
    image: rabbitmq:3.13-management
    ports:
      - "5672:5672"
      - "15672:15672"   # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: agents
      RABBITMQ_DEFAULT_PASS: agents_pass
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: agent_platform
      POSTGRES_USER: platform
      POSTGRES_PASSWORD: platform_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

**Deliverables:**
- [ ] `MessageBus` abstraction with RabbitMQ and Redis implementations
- [ ] `Blackboard` shared context store with Redis backend
- [ ] `DebateManager` with configurable consensus strategies
- [ ] docker-compose with RabbitMQ, Redis, and PostgreSQL
- [ ] Integration tests: two agents exchange messages through RabbitMQ

---

## Phase 5: Specialized Agents

**Duration:** 5--6 days
**Objective:** Implement five fully functional agents, each with a distinct persona,
tool set, and output schema. Wire them into the code review pipeline.

### Task 5.1: Planner Agent

The Planner takes a high-level goal and produces a structured execution plan.

```python
# src/agents/planner.py
from typing import Any
from src.agents.base import BaseAgent
from src.agents.registry import AgentRegistry


@AgentRegistry.register("planner")
class PlannerAgent(BaseAgent):
    """Decomposes complex tasks into actionable steps for other agents."""

    async def _execute(self, conversation: list[dict]) -> dict[str, Any]:
        response = await self._call_llm(conversation)

        return {
            "plan": response["steps"],
            "reasoning": response["reasoning"],
            "estimated_complexity": response["complexity"],
            "parallel_groups": response.get("parallel_groups", []),
            "risk_factors": response.get("risks", []),
        }

    def _build_conversation(self, message) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Task: {message.content['task']}\n\n"
                    f"Available agents: planner, coder, reviewer, tester, pm\n"
                    f"Context: {message.content.get('shared_context', {})}\n\n"
                    "Produce a step-by-step plan. For each step, specify:\n"
                    "- which agent should do it\n"
                    "- what their input is\n"
                    "- what their expected output is\n"
                    "- which steps can run in parallel\n"
                ),
            },
        ]
```

### Task 5.2: Coder Agent

The Coder writes, modifies, and refactors code based on task descriptions or review
feedback. It has access to file management and code execution tools.

### Task 5.3: Reviewer Agent

The Reviewer performs thorough code reviews using the structured output schema defined
in Phase 1 (Task 1.3). It reads files via GitHub API, checks for security issues, and
scores its own confidence on each finding.

### Task 5.4: Tester Agent

The Tester writes and runs tests for the code produced by the Coder. It has access to
a sandboxed code execution tool and can interpret test output.

### Task 5.5: PM Agent

The PM acts as a supervisor: it tracks overall progress, ensures agents stay on task,
resolves disputes by escalating or using the debate mechanism, and produces summary
reports for humans.

### Task 5.6: Research Team Variant

Implement the alternative persona set for the research workflow:

| Agent | Role | Model | Key Tools |
|-------|------|-------|-----------|
| Researcher | Find sources and raw information | Claude Opus | Web search, PDF reader |
| Analyst | Extract insights and patterns | GPT-4o | Data analysis, chart generation |
| Writer | Produce draft content | Claude Sonnet | Document editor |
| Editor | Improve clarity, structure, tone | GPT-4o | Grammar check, readability scorer |
| Fact-Checker | Verify claims against sources | Claude Haiku | Source search, citation verifier |

**Deliverables:**
- [ ] Five agent implementations (Planner, Coder, Reviewer, Tester, PM)
- [ ] Research team variant (Researcher, Analyst, Writer, Editor, Fact-Checker)
- [ ] YAML configs for all agents
- [ ] System prompt templates for all agents
- [ ] Integration test: run the code review pipeline end to end with a real PR

---

## Phase 6: Human-in-the-Loop

**Duration:** 3--4 days
**Objective:** Add human oversight controls so that critical decisions require approval,
agents can escalate uncertainty, and humans can override any agent output.

### Task 6.1: Escalation Triggers

Define rules that automatically escalate to a human. These are evaluated after every
agent step.

```python
# src/hitl/escalation.py
from typing import Any
from enum import Enum
import structlog

logger = structlog.get_logger()


class EscalationReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    SECURITY_FINDING = "security_finding"
    DISAGREEMENT = "agent_disagreement"
    COST_THRESHOLD = "cost_threshold"
    EXPLICIT_REQUEST = "agent_explicit_request"
    LOOP_DETECTED = "loop_detected"


class EscalationManager:
    """Evaluates whether a workflow step requires human intervention."""

    def __init__(self, rules: list[dict[str, Any]]):
        self.rules = rules

    def evaluate(
        self,
        agent_role: str,
        result: dict[str, Any],
        workflow_context: dict[str, Any],
    ) -> EscalationReason | None:
        """Check all rules and return the first triggered reason, or None."""
        for rule in self.rules:
            if rule.get("agent") and rule["agent"] != agent_role:
                continue

            reason = self._check_rule(rule, result, workflow_context)
            if reason:
                logger.info(
                    "escalation.triggered",
                    agent=agent_role,
                    reason=reason,
                    rule=rule["name"],
                )
                return reason

        return None

    def _check_rule(
        self,
        rule: dict[str, Any],
        result: dict[str, Any],
        context: dict[str, Any],
    ) -> EscalationReason | None:
        rule_type = rule["type"]

        if rule_type == "confidence_threshold":
            confidence = result.get("confidence", 1.0)
            if confidence < rule["threshold"]:
                return EscalationReason.LOW_CONFIDENCE

        elif rule_type == "severity_match":
            findings = result.get("findings", [])
            for finding in findings:
                if finding.get("severity") in rule["severities"]:
                    return EscalationReason.SECURITY_FINDING

        elif rule_type == "cost_limit":
            total_cost = context.get("total_cost_usd", 0)
            if total_cost > rule["max_usd"]:
                return EscalationReason.COST_THRESHOLD

        elif rule_type == "loop_detection":
            step_history = context.get("step_history", [])
            if self._detect_loop(step_history, rule.get("max_repeats", 3)):
                return EscalationReason.LOOP_DETECTED

        return None

    def _detect_loop(self, history: list[str], max_repeats: int) -> bool:
        """Detect if agents are cycling through the same steps."""
        if len(history) < max_repeats:
            return False
        recent = history[-max_repeats:]
        return len(set(recent)) == 1
```

### Task 6.2: Approval Gates

Certain workflow steps require explicit human approval before proceeding. The system
pauses the workflow and notifies the human via the dashboard and optionally Slack/email.

### Task 6.3: Override Mechanisms

Humans can override any agent decision at any point --- modify the output, skip a step,
or force a re-run with additional context.

### Task 6.4: Feedback Collection

When humans intervene, their corrections become training data for improving agent
prompts and routing rules over time.

**Deliverables:**
- [ ] `EscalationManager` with configurable rule engine
- [ ] Approval gate middleware that pauses workflows
- [ ] Override API endpoints (modify result, skip step, re-run)
- [ ] Feedback storage and retrieval for prompt improvement
- [ ] Tests: escalation triggers correctly on low confidence, security findings

---

## Phase 7: Evaluation

**Duration:** 3--4 days
**Objective:** Build a comprehensive evaluation framework that measures task completion,
individual agent contributions, conversation quality, and cost efficiency.

### Task 7.1: End-to-End Task Completion Scoring

Measure whether the workflow achieved its goal (e.g., "Was the code review thorough?
Were all bugs found?"). Use a rubric-based approach with an evaluator LLM.

### Task 7.2: Per-Agent Contribution Scoring

Track each agent's contribution: tokens used, tools called, quality of output,
how often their work was revised by downstream agents.

```python
# src/evaluation/agent_scoring.py
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentScorecard:
    agent_role: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_tool_calls: int = 0
    average_confidence: float = 0.0
    times_output_revised: int = 0
    escalations_triggered: int = 0
    average_latency_ms: float = 0.0
    cost_usd: float = 0.0
    quality_scores: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 0.0

    @property
    def average_quality(self) -> float:
        return sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0

    @property
    def cost_per_task(self) -> float:
        return self.cost_usd / self.tasks_completed if self.tasks_completed > 0 else 0.0
```

### Task 7.3: Conversation Quality Analysis

Measure the quality of inter-agent conversations: Are messages clear? Do agents build on
each other's work or repeat information? Is the debate productive?

### Task 7.4: Cost Analysis

Track token usage and API costs per agent, per workflow, and per workflow type. Identify
which agents are cost-efficient and which need prompt optimization.

**Deliverables:**
- [ ] Task completion scorer with rubric-based LLM evaluation
- [ ] `AgentScorecard` with per-agent metrics
- [ ] Conversation quality analyzer
- [ ] Cost tracking and reporting
- [ ] Dashboard integration: display evaluation results per workflow

---

## Phase 8: API & Dashboard

**Duration:** 5--6 days
**Objective:** Build the FastAPI backend for workflow management and a React/Next.js
dashboard with real-time visualization of agent collaboration.

### Task 8.1: Workflow Management API

```python
# src/api/routes/workflows.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


class WorkflowCreateRequest(BaseModel):
    template: str                     # "code_review" or "research_team"
    context: dict                     # template-specific parameters
    human_approval_required: bool = True


class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    steps: list[dict]
    created_at: str


@router.post("/", response_model=WorkflowResponse)
async def create_workflow(request: WorkflowCreateRequest):
    """Create and start a new multi-agent workflow."""
    ...


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str):
    """Get the current state of a workflow."""
    ...


@router.post("/{workflow_id}/steps/{step_id}/approve")
async def approve_step(workflow_id: str, step_id: str):
    """Approve a step that is waiting for human approval."""
    ...


@router.post("/{workflow_id}/steps/{step_id}/override")
async def override_step(workflow_id: str, step_id: str, body: dict):
    """Override an agent's output with human-provided content."""
    ...
```

### Task 8.2: WebSocket Real-Time Events

Stream agent messages and workflow state changes to the dashboard in real time.

```python
# src/api/routes/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Any
import json

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active: dict[str, list[WebSocket]] = {}  # workflow_id -> connections

    async def connect(self, workflow_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active.setdefault(workflow_id, []).append(websocket)

    async def broadcast(self, workflow_id: str, event: dict[str, Any]):
        for ws in self.active.get(workflow_id, []):
            await ws.send_json(event)


manager = ConnectionManager()


@router.websocket("/ws/workflows/{workflow_id}")
async def workflow_events(websocket: WebSocket, workflow_id: str):
    await manager.connect(workflow_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client commands (e.g., request replay)
    except WebSocketDisconnect:
        manager.active.get(workflow_id, []).remove(websocket)
```

### Task 8.3: React Dashboard

Build the dashboard with three main views:

1. **Workflow List** --- All workflows with status badges and summary metrics
2. **Workflow Detail** --- Real-time agent interaction graph (React Flow), message log,
   step timeline, cost tracker
3. **Agent Analytics** --- Per-agent scorecards, contribution charts, cost breakdown

### Task 8.4: Agent Interaction Graph Visualization

Use React Flow or D3.js to render a live graph where nodes are agents and edges are
messages. Animate edge flow as messages pass in real time.

**Deliverables:**
- [ ] FastAPI app with workflow CRUD, step approval/override endpoints
- [ ] WebSocket endpoint streaming real-time workflow events
- [ ] React dashboard with workflow list, detail, and analytics views
- [ ] Agent interaction graph with animated message flow
- [ ] Integration: dashboard connected to live API and WebSocket

---

## Phase 9: Containerization

**Duration:** 2--3 days
**Objective:** Containerize all services with Docker and orchestrate them with
docker-compose for local development and single-machine deployment.

### Task 9.1: Multi-Stage Dockerfiles

Create optimized Dockerfiles for each service:

```dockerfile
# docker/Dockerfile.api
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir build && python -m build --wheel
RUN pip install --no-cache-dir dist/*.whl

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ src/
COPY configs/ configs/
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Task 9.2: Full docker-compose Stack

Orchestrate API, agent workers, dashboard, RabbitMQ, Redis, PostgreSQL, Jaeger,
Prometheus, and Grafana in a single docker-compose file.

### Task 9.3: Health Checks and Dependency Ordering

Add health checks so services wait for their dependencies to be ready.

**Deliverables:**
- [ ] Dockerfiles for API, agent worker, dashboard, tracing
- [ ] docker-compose.yaml with all services
- [ ] Health checks and startup ordering
- [ ] Verified: `docker-compose up` starts the entire platform

---

## Phase 10: Testing & CI/CD

**Duration:** 3--4 days
**Objective:** Build a comprehensive test suite and automated CI/CD pipeline.

### Task 10.1: Unit Tests

Test all core components in isolation:

- **Message schema** --- Serialization, validation, edge cases
- **State machine** --- Step transitions, ready-step detection, deadlock detection
- **Router** --- Conditional branching with various result payloads
- **Decomposer** --- Template-based and LLM-based decomposition
- **Escalation rules** --- All trigger types and edge cases
- **Blackboard** --- Read/write/query operations

### Task 10.2: Integration Tests

Test component interactions with real infrastructure:

- **Agent communication** --- Two agents exchange messages via RabbitMQ
- **Workflow execution** --- Run a 3-step workflow with mock LLM responses
- **HITL flow** --- Escalation triggers, workflow pauses, human approves

### Task 10.3: End-to-End Tests

- **Code review pipeline** --- Submit a PR, agents review and produce a report
- **Research team** --- Submit a research question, agents produce a document

### Task 10.4: CI/CD Pipeline

```yaml
# .github/workflows/ci.yaml
name: CI
on:
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff mypy
      - run: ruff check src/ tests/
      - run: mypy src/ --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v --cov=src
      - run: pytest tests/integration/ -v

  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker compose build
```

**Deliverables:**
- [ ] Unit tests with >80% coverage on core modules
- [ ] Integration tests with real Redis/PostgreSQL
- [ ] End-to-end test for code review pipeline
- [ ] GitHub Actions CI: lint, test, build
- [ ] CD workflow for deploying to staging

---

## Phase 11: Monitoring & Observability

**Duration:** 3--4 days
**Objective:** Implement distributed tracing, metrics collection, and dashboards to
understand agent behavior, debug failures, and track costs.

### Task 11.1: OpenTelemetry Distributed Tracing

Every agent call, tool execution, and message exchange gets a trace span. This creates
a full timeline of how agents collaborated to complete a workflow.

```python
# src/monitoring/tracing.py
from functools import wraps
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource


def setup_tracing(service_name: str) -> None:
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint="http://jaeger:4317", insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


tracer = trace.get_tracer("agent-platform")


def trace_agent_call(func):
    """Decorator that creates a trace span for agent processing."""
    @wraps(func)
    async def wrapper(self, message, *args, **kwargs):
        with tracer.start_as_current_span(
            f"agent.{self.role}.process",
            attributes={
                "agent.name": self.name,
                "agent.role": self.role,
                "agent.model": self.model_id,
                "workflow.id": message.workflow_id,
                "message.type": message.message_type,
                "message.sender": message.sender,
            },
        ) as span:
            result = await func(self, message, *args, **kwargs)
            span.set_attribute("agent.output_size", len(str(result)))
            return result
    return wrapper
```

### Task 11.2: Prometheus Metrics

Define custom metrics for agent performance, cost, and system health.

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Agent-level metrics
agent_calls_total = Counter(
    "agent_calls_total",
    "Total agent invocations",
    ["agent_role", "model", "status"],
)

agent_latency_seconds = Histogram(
    "agent_latency_seconds",
    "Agent processing time",
    ["agent_role"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120],
)

agent_tokens_total = Counter(
    "agent_tokens_total",
    "Total tokens consumed",
    ["agent_role", "direction"],  # direction: input/output
)

agent_cost_usd = Counter(
    "agent_cost_usd_total",
    "Total API cost in USD",
    ["agent_role", "model"],
)

# Workflow-level metrics
workflow_duration_seconds = Histogram(
    "workflow_duration_seconds",
    "End-to-end workflow duration",
    ["template"],
)

workflow_steps_total = Counter(
    "workflow_steps_total",
    "Total workflow steps executed",
    ["template", "status"],
)

escalations_total = Counter(
    "escalations_total",
    "Human escalations triggered",
    ["reason", "agent_role"],
)

# System-level metrics
active_workflows = Gauge(
    "active_workflows",
    "Currently running workflows",
)

message_queue_depth = Gauge(
    "message_queue_depth",
    "Messages waiting in queue",
    ["queue_name"],
)
```

### Task 11.3: Agent Interaction Graphs

Build a post-hoc analysis tool that reconstructs the full collaboration graph for a
workflow: which agents talked to which, how many messages, what the information flow
looked like.

### Task 11.4: Grafana Dashboards

Create pre-configured dashboards:

1. **Platform Overview** --- Active workflows, success rate, cost per day, queue depth
2. **Agent Performance** --- Per-agent latency, token usage, success rate, cost
3. **Workflow Deep Dive** --- Timeline, agent interaction graph, step status, cost breakdown
4. **Alerts** --- Escalation rate spikes, cost anomalies, high error rates

### Task 11.5: Failure Analysis

Implement automated failure analysis that categorizes workflow failures:

- **Agent timeout** --- Which agent, how long, what was the task
- **Hallucination** --- Output validation failures
- **Loop detection** --- Agents sending each other back and forth
- **Cost overrun** --- Workflow exceeded budget before completing

**Deliverables:**
- [ ] OpenTelemetry setup with Jaeger exporter
- [ ] `trace_agent_call` decorator applied to all agents
- [ ] Prometheus metrics for agents, workflows, and system
- [ ] Four Grafana dashboards (overview, agent, workflow, alerts)
- [ ] Failure analysis categorization and reporting
- [ ] Verified: trace a full workflow from submission to completion in Jaeger

---

## Skills Checklist

By completing this project, you will have demonstrated:

| Skill | Where You Used It |
|-------|-------------------|
| **LLM API Integration** | Multiple providers (Anthropic, OpenAI) with function calling |
| **Multi-Agent Orchestration** | Sequential, parallel, hierarchical patterns |
| **Agent Communication Protocols** | Message schemas, pub/sub, blackboard pattern |
| **Distributed Systems** | Message queues, state machines, consensus |
| **Workflow State Management** | PostgreSQL-backed state machine with recovery |
| **Human-in-the-Loop Design** | Escalation, approval gates, override mechanisms |
| **Async Python** | asyncio, aio-pika, async generators |
| **API Design** | FastAPI, WebSocket, REST, Pydantic schemas |
| **Real-Time UI** | React, WebSocket, animated graph visualization |
| **Observability** | OpenTelemetry distributed tracing, Prometheus, Grafana |
| **Cost Management** | Per-agent and per-workflow cost tracking and budgets |
| **Evaluation** | LLM-as-judge, rubric-based scoring, quality metrics |
| **Containerization** | Docker multi-stage builds, docker-compose orchestration |
| **CI/CD** | GitHub Actions with multi-service test environments |
| **Prompt Engineering** | System prompts, structured output, few-shot examples |
| **Testing** | Unit, integration, and end-to-end with mock LLM responses |
