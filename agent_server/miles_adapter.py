"""Miles integration: launch the Router and the replicas, wait for them, and run one episode per sample.

Miles reaches this module only when ``--use-agent-server`` is set: from the process specs
(``miles/ray/specs/entrypoint.py``) and from the two rollout readiness points next to
``wait_session_server_ready``. Rollout uses
``--custom-generate-function-path agent_server.miles_adapter.generate``.
"""

import asyncio
import json
import logging
import os
import shlex
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx

from agent_server.router import RouterConfig
from agent_server.server import ReplicaConfig
from miles.ray.specs.inference import compute_router_pool_id
from miles.rollout.agentic.agent_function import call_agent_function
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.agentic_tool_call import build_chat_request_kwargs
from miles.rollout.generate_hub.agentic_tool_call import generate as session_server_generate
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2
from miles.rollout.session.types import CreateSessionRequest
from miles.utils.function_registry import load_function
from miles.utils.http_utils import post, wait_tcp_ready_async
from miles.utils.types import Sample
from miles.utils.workers.argv_utils import config_to_argv, python_argv_prefix
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import CommandWorkerSpec, LaunchCommandContext, PortInfo, SchedulingSpec

logger = logging.getLogger(__name__)

AGENT_SERVER_ROUTER_POOL_ID = "agent-server-router"
AGENT_SERVER_REPLICA_POOL_ID = "agent-server-replica"

_READY_TIMEOUT_SECONDS = 300.0
# Optional JSON object {"<replica port>": "<url>"}: the URL agents outside the cluster use for the
# replica on that port, e.g. a tunnel opened before the run starts. Needs --agent-server-port.
PUBLIC_URLS_ENV = "AGENT_SERVER_PUBLIC_URLS"
# agent_server/ sits next to the miles package, not inside it, so the spawned processes need the
# repository root on their path for ``python -m agent_server.*``.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------


def specs_agent_server(args) -> list[CommandWorkerSpec]:
    """One Router plus ``--agent-server-workers`` replicas, started and stopped by Miles' worker manager."""
    return [_spec_router(args), _spec_replicas(args)]


def agent_server_router_provider(capability: BackendCapability) -> BaseWorkerProvider:
    return capability.static_worker_provider(pool_id=AGENT_SERVER_ROUTER_POOL_ID)


async def wait_agent_server_ready(args, *, provider: BaseWorkerProvider) -> None:
    """Block until the Router is up and every replica has registered, then publish the Router URL on args."""
    router = (await provider.get_addrs(worker_name=compute_worker_name(pool_id=AGENT_SERVER_ROUTER_POOL_ID)))[
        "primary"
    ]
    await wait_tcp_ready_async(router.host, router.port, timeout=_READY_TIMEOUT_SECONDS)
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    async with httpx.AsyncClient() as client:
        while True:
            replicas = (await client.get(f"{router.addr}/replicas")).json()["replicas"]
            if len(replicas) >= args.agent_server_workers:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"only {len(replicas)} of {args.agent_server_workers} agent server replicas registered with {router.addr}"
                )
            await asyncio.sleep(1)
    # generate() reads this to create sessions.
    args.agent_server_router_url = router.addr
    logger.info("Agent server router ready at %s with replicas %s", router.addr, replicas)


def _spec_router(args) -> CommandWorkerSpec:
    def _launch_command(ctx: LaunchCommandContext) -> str:
        primary = ctx.self_addrs["primary"]
        config = RouterConfig(host=primary.host, port=primary.port, timeout=args.miles_router_timeout)
        return shlex.join([*python_argv_prefix(), "-m", "agent_server.router", *config_to_argv(config)])

    return CommandWorkerSpec(
        name=AGENT_SERVER_ROUTER_POOL_ID,
        port_infos=[PortInfo(name="primary", static_port=8000, allow_dynamic=True)],
        env_var=_env_var,
        scheduling=_cpu_scheduling(args, num_cells=1),
        launch_command=_launch_command,
    )


def _spec_replicas(args) -> CommandWorkerSpec:
    def _launch_command(ctx: LaunchCommandContext) -> str:
        primary = ctx.self_addrs["primary"]
        (inference_router,) = ctx.pool_addrs[compute_router_pool_id(0)]
        (router,) = ctx.pool_addrs[AGENT_SERVER_ROUTER_POOL_ID]
        public_urls = json.loads(os.environ.get(PUBLIC_URLS_ENV) or "{}")
        config = ReplicaConfig(
            host=primary.host,
            port=primary.port,
            router_url=router["primary"].addr,
            public_url=public_urls.get(str(primary.port)),
            backend_url=inference_router["primary"].addr,
            hf_checkpoint=args.hf_checkpoint,
            chat_template_path=args.chat_template_path,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            timeout=args.miles_router_timeout,
            use_rollout_routing_replay=args.use_rollout_routing_replay,
            use_addition_r3=getattr(args, "pause_generation_mode", None) != "retract",
            num_layers=getattr(args, "num_layers", None),
            session_sample_picker_path=args.session_sample_picker_path,
            session_sample_postprocessor_path=args.session_sample_postprocessor_path,
        )
        return shlex.join([*python_argv_prefix(), "-m", "agent_server.server", *config_to_argv(config)])

    if args.agent_server_port is None:
        port_info = PortInfo(name="primary", static_port=8000, allow_dynamic=True)
    else:
        port_info = PortInfo(name="primary", static_port=args.agent_server_port, offset_by_cell=True)
    return CommandWorkerSpec(
        name=AGENT_SERVER_REPLICA_POOL_ID,
        port_infos=[port_info],
        env_var=_env_var,
        scheduling=_cpu_scheduling(args, num_cells=args.agent_server_workers),
        launch_command=_launch_command,
    )


def _cpu_scheduling(args, *, num_cells: int) -> SchedulingSpec:
    return SchedulingSpec(
        num_cells=num_cells,
        num_workers_per_cell=1,
        num_gpus_per_worker=0,
        num_cpus_per_worker=0,
        pin_to_head=args.pin_rollout_manager_to_head,
    )


def _env_var(_ctx) -> dict[str, str]:
    return {"PYTHONPATH": os.pathsep.join(filter(None, [_REPO_ROOT, os.environ.get("PYTHONPATH")]))}


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Run one agent episode on a fresh agent server session and return its training samples.

    The session is created through the agent server's Router and the agent talks to the owning replica.
    ``OpenAIEndpointTracer`` then collects the samples from that replica, deletes the session and
    decodes the reply with Miles' samples codec.
    """
    args = input.args
    assert not args.partial_rollout, "Partial rollout is not supported"
    handle = await _create_session(
        args.agent_server_router_url, evaluation=input.evaluation, sampling_params=input.sampling_params
    )
    tracer = OpenAIEndpointTracer(
        router_url=handle["replica_url"], session_id=handle["session_id"], samples_wire_fields=COMPUTED_FIELDS_V2
    )
    log_prefix = f"[agent server session={tracer.session_id}]"

    max_seq_len = getattr(args, "max_seq_len", None)
    metadata = {**input.sample.metadata, "session_server_id": tracer.session_server_id}
    if max_seq_len is not None:
        metadata["max_seq_len"] = max_seq_len

    agent_metadata = None
    try:
        agent_metadata = await call_agent_function(
            load_function(args.custom_agent_function_path),
            mode=args.custom_agent_function_mode,
            base_url=handle["agent_base_url"],
            prompt=input.sample.prompt,
            request_kwargs=build_chat_request_kwargs(input.sampling_params),
            metadata=metadata,
        )
    except Exception as exc:
        # Still collect the turns the agent made before it failed.
        logger.warning("%s Agent function failed: %s", log_prefix, exc, exc_info=True)

    try:
        result = await tracer.collect_samples(input.sample, max_seq_len=max_seq_len, agent_metadata=agent_metadata)
    except (TimeoutError, httpx.TransportError) as exc:
        # Costs this sample, not the run; a non-2xx reply still raises.
        logger.warning("%s Failed collecting samples: %r", log_prefix, exc, exc_info=True)
        result = None
    if result is None or not result.samples:
        if result is not None:
            logger.warning("%s No samples collected (%s)", log_prefix, result.empty_reason)
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=[sample])

    samples = result.samples
    if len(samples) > 1:
        # All branches of one episode are counted as one rollout.
        rollout_id = input.sample.rollout_id if input.sample.rollout_id is not None else input.sample.index
        for sample in samples:
            sample.rollout_id = rollout_id
    tool_time = ((agent_metadata or {}).get("agent_metrics") or {}).get("total_tool_time")
    if tool_time is not None:
        for sample in samples:
            sample.non_generation_time = tool_time
    return GenerateFnOutput(samples=samples)


# Registers --custom-agent-function-path, --custom-agent-function-mode and --max-seq-len.
generate.add_arguments = session_server_generate.add_arguments


async def _create_session(router_url: str, *, evaluation: bool, sampling_params: dict | None) -> dict[str, Any]:
    # The sampling fields a session may default, plus evaluation.
    session_params = {
        key: value for key, value in (sampling_params or {}).items() if key in CreateSessionRequest.model_fields
    }
    body = CreateSessionRequest.model_validate({**session_params, "evaluation": evaluation})
    return await post(f"{router_url}/sessions", body.model_dump(exclude_none=True))
