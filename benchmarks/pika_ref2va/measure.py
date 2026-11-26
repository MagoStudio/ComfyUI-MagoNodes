"""Time the usual and Sol3 sampler nodes on a local ComfyUI server."""

import argparse
import asyncio
import json
import secrets
import sys
import time
import uuid
from pathlib import Path

import aiohttp


SAMPLERS = {"13": "usual", "113": "Sol3"}
PROMPT = Path(__file__).with_name("prompt.api.json")
# H3's native frame counts used in the recorded runs, before trimming to 24 fps.
NATIVE_FRAMES = {5: 124, 10: 243, 15: 362}


def load_graph(seconds, reference_image=None):
    graph = json.loads(PROMPT.read_text(encoding="utf-8"))
    graph["8"]["inputs"]["length"] = NATIVE_FRAMES[seconds]
    for frames, audio, save, name in (("33", "35", "17", "usual"), ("133", "135", "117", "sol3")):
        graph[frames]["inputs"]["length"] = 24 * seconds
        graph[audio]["inputs"]["duration"] = seconds
        graph[save]["inputs"]["filename_prefix"] = f"benchmarks/pika_ref2va_720p{seconds}s/{name}"
    if reference_image is not None:
        graph["7"]["inputs"]["image"] = reference_image
    return graph


async def check_server(session, url, graph):
    async with session.get(f"{url}/object_info") as response:
        response.raise_for_status()
        info = await response.json()
    missing = {node["class_type"] for node in graph.values()} - info.keys()
    if missing:
        raise RuntimeError(f"Missing ComfyUI node types: {', '.join(sorted(missing))}")
    if not any("dense_first_blocks" in fields for fields in info["MagoSol3Attention"]["input"].values()):
        raise RuntimeError("Restart ComfyUI with the updated Sol3 node.")
    for node_id, field in (("1", "unet_name"), ("2", "lora_name"), ("4", "clip_name"),
                           ("5", "vae_name"), ("6", "vae_name")):
        node = graph[node_id]
        filename = node["inputs"][field]
        choices = info[node["class_type"]]["input"]["required"][field][0]
        if filename not in choices:
            raise RuntimeError(f"{node['class_type']} cannot find {filename}.")
    image = Path(graph["7"]["inputs"]["image"])
    params = {"type": "input", "filename": image.name, "subfolder": image.parent.as_posix()}
    async with session.head(f"{url}/view", params=params) as response:
        if response.status == 404:
            raise RuntimeError(f"Missing ComfyUI input image: {image}. Upload it or pass --reference-image.")
        response.raise_for_status()


async def run_pair(session, url, graph, seed, label):
    graph["9"]["inputs"]["noise_seed"] = seed
    client_id = uuid.uuid4().hex
    async with session.ws_connect(f"{url}/ws", params={"clientId": client_id}) as ws:
        async with session.post(f"{url}/prompt", json={"client_id": client_id, "prompt": graph}) as response:
            body = await response.json()
            if response.status != 200 or body.get("node_errors"):
                raise RuntimeError(f"ComfyUI rejected the benchmark: {body}")
        prompt_id = body["prompt_id"]
        print(f"{label}, seed {seed}, job {prompt_id}", flush=True)
        starts, times, cached = {}, {}, set()
        async for message in ws:
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(message.data)
            data = event.get("data", {})
            if data.get("prompt_id") != prompt_id:
                continue
            kind = event["type"]
            if kind == "progress_state":
                now = time.monotonic()
                for node, name in SAMPLERS.items():
                    state = data["nodes"].get(node, {}).get("state")
                    if state == "running" and node not in starts:
                        starts[node] = now
                    elif state == "finished" and node in starts and node not in times:
                        times[node] = now - starts[node]
                        print(f"  {name} sampler: {times[node]:.2f} s", flush=True)
            elif kind == "executed" and graph[data["node"]]["class_type"] == "SaveVideo":
                for image in data["output"].get("images", []):
                    path = Path("output", image.get("subfolder", ""), image["filename"])
                    print(f"  Saved {path}", flush=True)
            elif kind == "execution_cached":
                cached.update(SAMPLERS.keys() & set(data["nodes"]))
            elif kind in ("execution_error", "execution_interrupted"):
                raise RuntimeError(f"{kind} in job {prompt_id}, node {data.get('node_id')}: "
                                   f"{data.get('exception_message', 'interrupted')}")
            elif kind == "execution_success":
                if cached:
                    raise RuntimeError(f"Cached sampler(s) {', '.join(sorted(cached))} in job {prompt_id}. "
                                       "Rerun with a fresh --seed to obtain timings.")
                if times.keys() != SAMPLERS.keys():
                    raise RuntimeError(f"Incomplete sampler timings for job {prompt_id}: {times}")
                return times
    raise RuntimeError(f"WebSocket disconnected. Check ComfyUI history for job {prompt_id}.")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--seconds", type=int, choices=NATIVE_FRAMES, default=5,
                        help="Output clip duration at 24 fps (default: 5).")
    parser.add_argument("--reference-image", help="Existing image filename relative to ComfyUI's input folder.")
    parser.add_argument("--seed", type=int, help="First pair's seed. By default each run picks a fresh seed.")
    parser.add_argument("--no-warmup", action="store_true", help="Measure one pair without a warmup run.")
    parser.add_argument("--check", action="store_true",
                        help="Check the paired samplers, installed nodes, model files and reference image without queuing.")
    args = parser.parse_args()
    graph = load_graph(args.seconds, args.reference_image)
    if graph["9"]["class_type"] != "RandomNoise" or any(
        graph[n]["class_type"] != "SamplerCustomAdvanced" or graph[n]["inputs"]["noise"] != ["9", 0]
        for n in SAMPLERS
    ):
        raise ValueError("The paired sampler nodes must both use RandomNoise node 9.")
    seed = secrets.randbits(52) if args.seed is None else args.seed
    if not 0 <= seed <= 2**64 - 1 - int(not args.no_warmup):
        parser.error("The seeds for both pairs must fit in an unsigned 64-bit integer.")
    url = f"http://127.0.0.1:{args.port}"
    async with aiohttp.ClientSession() as session:
        if args.check:
            await check_server(session, url, graph)
            print(f"{args.seconds}s pair ready: shared noise, nodes, Sol3 block control, model files and input image found.")
            return
        if not args.no_warmup:
            await run_pair(session, url, graph, seed, "Warmup")
            seed += 1
        times = await run_pair(session, url, graph, seed, "Measured pair")
        print(f"Measured sampler speedup (usual / Sol3): {times['13'] / times['113']:.3f}x")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (aiohttp.ClientError, OSError, RuntimeError, ValueError) as exc:
        sys.exit(str(exc))
