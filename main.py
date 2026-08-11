import os
import sys
import json
import asyncio
import threading
import time
import queue
import h5py
import pandas as pd
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import uvicorn
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

try:
    import torch
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except ImportError:
    pass

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    MOCK_MODE = False
    print("LIBERO environment loaded successfully.")
except ImportError as e:
    print(f"Warning: Libero not found → MOCK mode. ({e})")
    MOCK_MODE = True

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
os.makedirs("static", exist_ok=True)
app.mount("/dashboard", StaticFiles(directory="static", html=True), name="static")

executor = ThreadPoolExecutor(max_workers=1)

# ─── Shared Application State ─────────────────────────────────────────────────
# NOTE: action_chunks is NOT in broadcast state (too large for WebSocket)
# It is served via GET /api/chunks instead.
_action_chunks_store: list = []   # internal store, not broadcast

state = {
    "sim_running": False,
    "current_episode": 0,
    "current_frame": 0,
    "max_frames": 0,
    "chunk_progress": 0,
    "total_chunks": 0,
    "target_chunks": 0,
    "inference_point_3d": [0.0, 0.0, 0.0],
    "trajectory": [],
    "pipeline_stage": "IDLE",
    "data_flow_content": {},
    "log_messages": [],
    "successes": 0,
    "failures": 0,
    "total": 0,
    # Live raw data from parquet/HDF5 playback
    "current_action_vector": [],
    "current_obs_state": [],
    "current_task_language": "",
    "current_suite": "",
    "current_scenario": "",
    # Dataset browser
    "available_suites": [],
    "available_scenarios": [],
    "available_episodes": [],
    # Small summary for chart (downsampled if needed)
    "chunks_loaded": False,
    "chunks_count": 0,
    "episode_id": -1,
}

clients = []
global_frame = None
global_agent_frame = None
command_queue = queue.Queue()


def log(msg):
    print(msg)
    ts = time.strftime("%H:%M:%S")
    state["log_messages"].append(f"[{ts}] {msg}")
    if len(state["log_messages"]) > 100:
        state["log_messages"] = state["log_messages"][-100:]


# ─── Dataset Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(os.path.expanduser("~"), "LIBERO_3d_simulator_with_dataset")
CHUNK_DIR = os.path.join(REPO_ROOT, "lerobot_visualizer", "data", "chunk-000")
DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")

SUITE_PATHS = {
    "lerobot/libero_10 (Parquet)": CHUNK_DIR,
    "lerobot/libero_spatial (Parquet)": "/home/inc-kyh/VLAMARL/lerobot/datasets/libero_spatial_finetune/data/chunk-000",
    "libero_spatial (HDF5)": os.path.join(DATASETS_DIR, "libero_spatial"),
    "libero_object (HDF5)": os.path.join(DATASETS_DIR, "libero_object"),
    "libero_goal (HDF5)": os.path.join(DATASETS_DIR, "libero_goal"),
    "libero_100 (HDF5)": os.path.join(DATASETS_DIR, "libero_100"),
}


# ─── Parquet/HDF5 Replay Engine ────────────────────────────────────────────────
class ReplayEngine:
    def __init__(self):
        self.benchmark_dict = benchmark.get_benchmark_dict() if not MOCK_MODE else {}
        self.scenario_files = []
        self.current_scenario_path = None
        self.df_parquet = None
        self.episodes = []
        self.current_episode_id = 0
        self.current_episode_df = None
        self.current_frame = 0
        self.max_frames = 0
        self.env = None
        self.task = None
        self.task_suite = None
        self.task_suite_name = "libero_10"
        self.last_obs = None
        self.is_manual_mode = False
        self.last_action_val = [0.0] * 7

        # scan available suite files
        self._refresh_suites()

    def _refresh_suites(self):
        state["available_suites"] = list(SUITE_PATHS.keys())

    def refresh_suite_files(self, suite_name):
        target_dir = SUITE_PATHS.get(suite_name, CHUNK_DIR)
        files = []
        if os.path.exists(target_dir):
            for root, _, fnames in os.walk(target_dir):
                for f in fnames:
                    if f.endswith(".parquet") or f.endswith(".hdf5"):
                        files.append(os.path.join(root, f))
        self.scenario_files = sorted(files)
        state["available_scenarios"] = [os.path.basename(f) for f in self.scenario_files]
        state["current_suite"] = suite_name
        log(f"Suite '{suite_name}': found {len(self.scenario_files)} file(s).")
        if self.scenario_files:
            self.load_scenario_file(self.scenario_files[0])

    def load_scenario_file(self, file_path):
        def _exec():
            self.current_scenario_path = file_path
            if file_path.endswith(".parquet"):
                self.df_parquet = pd.read_parquet(file_path)
                self.episodes = sorted(self.df_parquet["episode_index"].unique().tolist())
            elif file_path.endswith(".hdf5"):
                with h5py.File(file_path, "r") as hf:
                    if "data" in hf:
                        demos = sorted(hf["data"].keys(), key=lambda x: int(x.split("_")[1]))
                        self.episodes = list(range(len(demos)))
                    else:
                        self.episodes = [0]
            state["available_episodes"] = self.episodes
            state["current_scenario"] = os.path.basename(file_path)
            log(f"Loaded scenario: {os.path.basename(file_path)}, {len(self.episodes)} episode(s)")
            self._select_episode_internal(self.episodes[0] if self.episodes else 0)
        executor.submit(_exec).result()

    def select_episode(self, episode_id):
        executor.submit(lambda: self._select_episode_internal(int(episode_id))).result()

    def _select_episode_internal(self, episode_id):
        self.current_episode_id = int(episode_id)
        if self.current_scenario_path and self.current_scenario_path.endswith(".parquet"):
            ep_df = self.df_parquet[self.df_parquet["episode_index"] == self.current_episode_id]
            self.current_episode_df = ep_df.sort_values("frame_index").reset_index(drop=True)
        elif self.current_scenario_path and self.current_scenario_path.endswith(".hdf5"):
            with h5py.File(self.current_scenario_path, "r") as hf:
                demo_key = f"demo_{self.current_episode_id}"
                if "data" in hf and demo_key in hf["data"]:
                    demo = hf["data"][demo_key]
                    actions = demo["actions"][:]
                    states = demo["states"][:] if "states" in demo else np.zeros((len(actions), 92))
                    self.current_episode_df = pd.DataFrame({
                        "action": list(actions),
                        "observation.state": list(states),
                        "frame_index": list(range(len(actions))),
                        "episode_index": [self.current_episode_id] * len(actions),
                        "task_index": [0] * len(actions),
                        "timestamp": [i * 0.1 for i in range(len(actions))],
                    })
                else:
                    self.current_episode_df = pd.DataFrame()
        self.max_frames = len(self.current_episode_df) if self.current_episode_df is not None else 0
        self.current_frame = 0
        state["current_episode"] = self.current_episode_id
        state["current_frame"] = 0
        state["max_frames"] = self.max_frames
        state["chunk_progress"] = 0
        state["target_chunks"] = self.max_frames
        state["trajectory"] = []

        # ── Populate real action chunks (internal store, NOT broadcast) ──
        if self.current_episode_df is not None and self.max_frames > 0:
            chunks = []
            for _, row in self.current_episode_df.iterrows():
                action = row.get("action", [0.0]*7)
                if hasattr(action, 'tolist'):
                    action = action.tolist()
                chunks.append([round(float(a), 4) for a in action])
            _action_chunks_store.clear()
            _action_chunks_store.extend(chunks)
        else:
            _action_chunks_store.clear()

        # Broadcast only lightweight summary
        state["chunks_loaded"] = len(_action_chunks_store) > 0
        state["chunks_count"] = len(_action_chunks_store)
        state["episode_id"] = self.current_episode_id
        state["target_chunks"] = self.max_frames
        state["total_chunks"] = self.max_frames

        log(f"Episode {episode_id} loaded ({self.max_frames} frames).")
        if not MOCK_MODE:
            self._load_env_for_task()

    def _load_env_for_task(self):
        task_idx = 0
        if self.current_episode_df is not None and len(self.current_episode_df) > 0:
            if "task_index" in self.current_episode_df.columns:
                task_idx = int(self.current_episode_df["task_index"].iloc[0])
        suite_name = self.task_suite_name
        try:
            self.task_suite = self.benchmark_dict[suite_name]()
            task_idx = min(task_idx, self.task_suite.n_tasks - 1)
            self.task = self.task_suite.get_task(task_idx)
            bddl_file = os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file)
            env_args = {
                "bddl_file_name": bddl_file,
                "camera_heights": 512, "camera_widths": 512,
                "camera_names": ["agentview", "robot0_eye_in_hand", "sideview"],
                "render_gpu_device_id": 0,
            }
            if self.env:
                try: self.env.close()
                except: pass
            self.env = OffScreenRenderEnv(**env_args)
            self.env.seed(0)
            self.last_obs = self.env.reset()
            init_states = self.task_suite.get_task_init_states(task_idx)
            if len(init_states) > 0:
                self.last_obs = self.env.set_init_state(init_states[0])
            # Warm up cameras
            self.env.sim.forward()
            for cam in ["agentview", "robot0_eye_in_hand", "sideview"]:
                try: self.env.sim.render(camera_name=cam, height=512, width=512)
                except: pass
            state["current_task_language"] = self.task.language
            log(f"Environment ready: {self.task.language}")
        except Exception as e:
            log(f"Env load error: {e}")

    def get_three_cameras(self):
        """Return side-by-side numpy image of 3 cameras (agentview|wrist|side)."""
        if MOCK_MODE:
            imgs = []
            labels = ["AGENT VIEW", "WRIST CAM", "SIDE VIEW"]
            colors = [(30, 80, 200), (200, 80, 30), (80, 200, 30)]
            for label, col in zip(labels, colors):
                img = np.full((512, 512, 3), col, dtype=np.uint8)
                cv2.putText(img, label, (120, 256), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
                cv2.putText(img, "MOCK MODE", (170, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
                imgs.append(img)
            return np.hstack(imgs)
        
        def _exec():
            imgs = []
            for cam in ["agentview", "robot0_eye_in_hand", "sideview"]:
                try:
                    obs = self.env.sim.render(camera_name=cam, height=512, width=512, depth=False)
                    img = np.ascontiguousarray(np.flipud(obs))
                except Exception:
                    img = np.zeros((512, 512, 3), dtype=np.uint8)
                imgs.append(img)
            return np.hstack(imgs)
        return executor.submit(_exec).result()

    def step_frame(self):
        """Advance one frame from dataset replay."""
        if self.current_episode_df is None or self.current_frame >= self.max_frames:
            return False
        row = self.current_episode_df.iloc[self.current_frame]
        action = row.get("action", [0.0]*7)
        if hasattr(action, 'tolist'):
            action = action.tolist()
        state["current_action_vector"] = [round(float(a), 4) for a in action]
        obs_s = row.get("observation.state", [])
        if hasattr(obs_s, 'tolist'):
            obs_s = obs_s.tolist()
        state["current_obs_state"] = [round(float(v), 4) for v in obs_s[:10]] if obs_s else []
        if not MOCK_MODE and self.env is not None:
            try:
                executor.submit(lambda: self.env.step(action)).result()
            except Exception as e:
                log(f"Step error: {e}")
        self.current_frame += 1
        state["current_frame"] = self.current_frame
        return True

    def teleop_step(self, action_vec):
        if MOCK_MODE:
            return
        def _exec():
            self.env.step(action_vec)
        executor.submit(_exec).result()

    def reset_env(self):
        if MOCK_MODE:
            state["current_frame"] = 0
            return
        def _exec():
            self.last_obs = self.env.reset()
            init_states = self.task_suite.get_task_init_states(0) if self.task_suite else []
            if len(init_states) > 0:
                self.last_obs = self.env.set_init_state(init_states[0])
            self.current_frame = 0
            state["current_frame"] = 0
        executor.submit(_exec).result()


replay = ReplayEngine()

# Scan default suite on startup
def init_suites():
    replay.refresh_suite_files("lerobot/libero_10 (Parquet)")
threading.Thread(target=init_suites, daemon=True).start()


# ─── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def get_stats():
    return {"successes": state["successes"], "failures": state["failures"], "total": state["total"]}

@app.get("/api/datasets")
async def get_datasets():
    return {
        "suites": state["available_suites"],
        "scenarios": state["available_scenarios"],
        "episodes": state["available_episodes"],
    }

@app.post("/api/select_suite")
async def select_suite(data: dict):
    suite = data.get("suite", "lerobot/libero_10 (Parquet)")
    def _run(): replay.refresh_suite_files(suite)
    threading.Thread(target=_run, daemon=True).start()
    log(f"Suite selected: {suite}")
    return {"status": "ok"}

@app.post("/api/select_scenario")
async def select_scenario(data: dict):
    scenario_name = data.get("scenario")
    match = [f for f in replay.scenario_files if os.path.basename(f) == scenario_name]
    if match:
        def _run(): replay.load_scenario_file(match[0])
        threading.Thread(target=_run, daemon=True).start()
        log(f"Scenario: {scenario_name}")
    return {"status": "ok"}

@app.post("/api/select_episode")
async def select_episode(data: dict):
    ep = data.get("episode", 0)
    def _run(): replay.select_episode(ep)
    threading.Thread(target=_run, daemon=True).start()
    log(f"Episode: {ep}")
    return {"status": "ok"}

@app.post("/api/control")
async def control(data: dict):
    command_queue.put(data)
    return {"status": "ok"}

@app.post("/api/teleop")
async def teleop(data: dict):
    command_queue.put({"type": "teleop", **data})
    return {"status": "ok"}

@app.get("/api/chunks")
async def get_chunks():
    """Return all action vectors for the loaded episode (served once via REST, not WebSocket)."""
    magnitudes = [
        round(float(np.sqrt(sum(a**2 for a in chunk))), 4)
        for chunk in _action_chunks_store
    ]
    return {
        "count": len(_action_chunks_store),
        "episode_id": state.get("episode_id", -1),
        "magnitudes": magnitudes,          # lightweight: one float per frame
        "vectors": _action_chunks_store,   # full 7-DOF (fetched once)
    }

@app.post("/api/execute_chunks")
async def execute_chunks(data: dict):
    target = data.get("target", 50)
    state["target_chunks"] = target
    # Run in dedicated thread so robot_loop is NOT blocked
    threading.Thread(
        target=_run_chunk_inference,
        args=(target,),
        daemon=True,
    ).start()
    return {"status": "ok", "target": target}

@app.post("/api/report_failure")
async def report_failure(data: dict):
    reason = data.get("reason", "Unknown")
    state["failures"] += 1
    state["total"] += 1
    log(f"Failure recorded: {reason}")
    return {"status": "ok"}

@app.post("/api/report_success")
async def report_success(data: dict):
    state["successes"] += 1
    state["total"] += 1
    log("Success recorded.")
    return {"status": "ok"}

@app.websocket("/ws/dataflow")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)

async def broadcast_state():
    while True:
        if clients:
            msg = json.dumps(state)
            dead = []
            for client in clients:
                try:
                    await client.send_text(msg)
                except:
                    dead.append(client)
            for d in dead:
                if d in clients: clients.remove(d)
        await asyncio.sleep(0.2)  # 5fps broadcast — smooth enough, much less CPU

@app.get("/api/stream")
async def video_stream():
    async def frame_generator():
        while True:
            if global_frame is not None:
                ret, buf = cv2.imencode('.jpg', global_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            await asyncio.sleep(0.03)
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stream/agent")
async def video_stream_agent():
    async def frame_generator():
        while True:
            if global_agent_frame is not None:
                ret, buf = cv2.imencode('.jpg', global_agent_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            await asyncio.sleep(0.03)
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# ─── Main Robot / Replay Loop ──────────────────────────────────────────────────
def robot_loop():
    log("Robot loop started.")
    sim_paused = True
    is_playing = False  # dataset replay auto-play

    while True:
        # Render 3-camera frame
        frame_rgb = replay.get_three_cameras()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        global global_frame, global_agent_frame
        global_frame = frame_bgr

        # Push agentview-only (left third) for chunk inspector
        # ONLY update this view during chunk inference (MUJOCO_STEP) or at reset (frame 0)
        # This prevents the inspector view from mirroring normal teleop/playback movements.
        if state.get("pipeline_stage") == "MUJOCO_STEP" or state.get("current_frame", 0) == 0:
            h, w = frame_bgr.shape[:2]
            global_agent_frame = frame_bgr[:, :w//3, :].copy()

        # Process commands
        try:
            cmd = command_queue.get_nowait()
        except queue.Empty:
            cmd = None

        if cmd:
            t = cmd.get("type") or cmd.get("action")
            if t == "sim_start":
                sim_paused = False
                log("Simulation started.")
                state["sim_running"] = True
            elif t == "sim_stop":
                sim_paused = True
                is_playing = False
                log("Simulation paused.")
                state["sim_running"] = False
            elif t == "sim_reset":
                replay.reset_env()
                sim_paused = True
                is_playing = False
                state["sim_running"] = False
                log("Environment reset.")
            elif t == "play":
                is_playing = True
                sim_paused = False
                state["sim_running"] = True
            elif t == "pause":
                is_playing = False
            elif t == "step_next":
                replay.step_frame()
            elif t == "seek":
                frame = cmd.get("frame", 0)
                replay.current_frame = int(frame)
                state["current_frame"] = replay.current_frame
            elif t == "teleop" or t == "move":
                action = [
                    float(cmd.get("dx", 0.0)),
                    float(cmd.get("dy", 0.0)),
                    float(cmd.get("dz", 0.0)),
                    float(cmd.get("droll", 0.0)),
                    float(cmd.get("dpitch", 0.0)),
                    float(cmd.get("dyaw", 0.0)),
                    float(cmd.get("gripper", replay.last_action_val[6] if len(replay.last_action_val) > 6 else 0.0)),
                ]
                replay.last_action_val = action
                replay.teleop_step(action)
                log(f"Teleop: {[round(a,3) for a in action]}")
            elif t == "gripper":
                val = float(cmd.get("value", 0.0))
                if len(replay.last_action_val) > 6:
                    replay.last_action_val[6] = val
                log(f"Gripper: {val}")
            elif t == "execute_chunks":
                pass  # now handled by dedicated thread via /api/execute_chunks

        # Auto-play dataset
        if is_playing and not sim_paused:
            ok = replay.step_frame()
            if not ok:
                is_playing = False
                state["sim_running"] = False
                log("Episode playback complete.")

        time.sleep(0.05)  # ~20fps


def _run_chunk_inference(target):
    """Replay the actual episode frames 0..target-1 in MuJoCo."""
    df = replay.current_episode_df
    if df is None or len(df) == 0:
        log("No episode loaded — cannot execute chunks.")
        return

    total = len(df)
    target = min(int(target), total)
    state["target_chunks"] = target
    state["chunk_progress"] = 0
    state["trajectory"] = []

    # Stage 1: Preprocessing
    state["pipeline_stage"] = "PREPROCESSING"
    state["data_flow_content"] = {
        "images": "Shape(1, 3, 224, 224)",
        "language": state.get("current_task_language", "—"),
        "state": f"Shape(1, {total})",
        "total_frames_in_episode": total,
    }
    time.sleep(0.3)

    # Stage 2: "Inference" — present the loaded action vectors
    pt_3d = [round(float(np.random.uniform(-0.2, 0.2)), 3),
             round(float(np.random.uniform(-0.2, 0.2)), 3),
             round(float(np.random.uniform(0.8, 1.2)), 3)]
    state["inference_point_3d"] = pt_3d
    state["pipeline_stage"] = "INFERENCE"
    state["data_flow_content"] = {
        "action_chunks_shape": f"({total}, 7)",
        "executing_up_to_frame": target,
        "extracted_3d_point": pt_3d,
    }
    time.sleep(0.3)

    # Stage 3: MuJoCo Step — replay actual episode actions
    state["pipeline_stage"] = "MUJOCO_STEP"
    # Reset env to episode initial state first
    replay.reset_env()
    replay.current_frame = 0
    state["current_frame"] = 0

    for i in range(target):
        row = df.iloc[i]
        action = row.get("action", [0.0]*7)
        if hasattr(action, 'tolist'):
            action = action.tolist()

        # Broadcast live action
        state["current_action_vector"] = [round(float(a), 4) for a in action]
        state["chunk_progress"] = i + 1
        state["current_frame"] = i + 1
        state["data_flow_content"] = {
            "frame": i,
            "action_vector": [round(float(a), 3) for a in action]
        }

        # Accumulate trajectory from end-effector position (obs state indices 0-2)
        obs_s = row.get("observation.state", [])
        if hasattr(obs_s, 'tolist'):
            obs_s = obs_s.tolist()
        if len(obs_s) >= 3:
            traj_pt = [round(float(obs_s[0]), 3),
                       round(float(obs_s[1]), 3),
                       round(float(obs_s[2]), 3)]
        else:
            traj_pt = pt_3d
        state["trajectory"].append(traj_pt)
        if len(state["trajectory"]) > 500:
            state["trajectory"].pop(0)

        # Step the simulation
        if not MOCK_MODE and replay.env is not None:
            try:
                a = action
                executor.submit(lambda a=a: replay.env.step(a)).result()
            except Exception as e:
                log(f"Step {i} error: {e}")

        replay.current_frame = i + 1
        time.sleep(0.05)  # ~20fps playback

    state["pipeline_stage"] = "IDLE"
    log(f"Chunk replay complete: {target}/{total} frames.")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_state())
    threading.Thread(target=robot_loop, daemon=True).start()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7052, log_level="info")

import re

def parse_dashboard_log(suite_name="libero_spatial"):
    try:
        if not os.path.exists("dashboard.log"): return {"total_success": 0, "total_episodes": 0, "tasks": []}
        with open("dashboard.log", "r", encoding="utf-8") as f:
            lines = f.readlines()
    except:
        return {"total_success": 0, "total_episodes": 0, "tasks": []}

    tasks_info = {}
    total_succ = 0
    total_eps = 0

    # Parse [robosuite INFO] Task N: <lang>
@app.get("/api/eval/tasks")
def get_eval_tasks(suite: str = "libero_spatial"):
    import json
    suite_clean = suite.replace('lerobot/', '').replace(' (Parquet)', '').replace(' (HDF5)', '')
    eval_path = f"/home/inc-kyh/VLAMARL/eval_logs/smolvla_full_eval_{suite_clean}/eval_info.json"
    rec_path = f"/home/inc-kyh/VLAMARL/eval_logs/dataset_recording_{suite_clean}/eval_info.json"
    
    target_path = eval_path if os.path.exists(eval_path) else rec_path
    if not os.path.exists(target_path):
        return {"total_success": 0, "total_episodes": 0, "tasks": []}
    
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    tasks = []
    total_succ = 0
    total_eps = 0
    
    try:
        from libero.libero.benchmark import get_benchmark
        b = get_benchmark(suite_clean)()
        tnames = b.get_task_names()
    except Exception as e:
        print("Benchmark error:", e)
        tnames = []

    for t in data.get("per_task", []):
        task_id = t.get("task_id")
        succs = t.get("metrics", {}).get("successes", [])
        n_succ = sum(1 for x in succs if x)
        n_tot = len(succs)
        total_succ += n_succ
        total_eps += n_tot
        
        tname = tnames[task_id] if task_id < len(tnames) else f"Task {task_id}"
        
        tasks.append({
            "task_id": task_id,
            "language": tname,
            "episodes": n_tot,
            "successes": n_succ
        })
        
    return {
        "total_success": total_succ,
        "total_episodes": total_eps,
        "tasks": tasks
    }

import glob
from fastapi.responses import FileResponse

def find_video_path(suite, task_id, ep_id):
    # Search for video in VLAMARL/eval_logs
    search_path = f"/home/inc-kyh/VLAMARL/eval_logs/*/videos/*{suite}*{task_id}/eval_episode_{ep_id}.mp4"
    matches = glob.glob(search_path)
    if matches: return matches[0]
    
    # Try alternate naming
    search_path2 = f"/home/inc-kyh/VLAMARL/eval_logs/*/videos/*{task_id}/eval_episode_{ep_id}.mp4"
    matches2 = glob.glob(search_path2)
    if matches2:
        for m in matches2:
            if suite in m: return m
    return None

@app.get("/api/eval/episodes")
def get_eval_episodes(suite: str = "libero_spatial", task_id: int = 0, success_only: bool = False):
    import json
    suite_clean = suite.replace('lerobot/', '').replace(' (Parquet)', '').replace(' (HDF5)', '')
    eval_path = f"/home/inc-kyh/VLAMARL/eval_logs/smolvla_full_eval_{suite_clean}/eval_info.json"
    rec_path = f"/home/inc-kyh/VLAMARL/eval_logs/dataset_recording_{suite_clean}/eval_info.json"
    
    target_path = eval_path if os.path.exists(eval_path) else rec_path
    if not os.path.exists(target_path):
        return {"total": 0, "episodes": []}
        
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    eps = []
    for t in data.get("per_task", []):
        if t.get("task_id") == task_id:
            succs = t.get("metrics", {}).get("successes", [])
            for ep_id, succ in enumerate(succs):
                if success_only and not succ: continue
                vid_path = find_video_path(suite_clean, task_id, ep_id)
                eps.append({"ep_id": ep_id, "success": bool(succ), "video_exists": vid_path is not None})
            break
            
    return {"total": len(eps), "episodes": eps}

@app.get("/api/eval/video")
def get_eval_video(suite: str, task_id: int, ep_id: int):
    suite_clean = suite.replace('lerobot/', '').replace(' (Parquet)', '').replace(' (HDF5)', '')
    vid_path = find_video_path(suite_clean, task_id, ep_id)
    if vid_path and os.path.exists(vid_path):
        return FileResponse(vid_path, media_type="video/mp4")
    return {"error": "Video not found"}
