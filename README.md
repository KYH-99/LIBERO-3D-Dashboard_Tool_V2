# LIBERO 3D Dashboard Tool V2

A comprehensive, interactive web dashboard for monitoring, evaluating, and visualizing Robosuite/LIBERO and LeRobot (Parquet) datasets in real-time. This tool bridges the gap between raw evaluation logs and intuitive visual analysis, enabling researchers to quickly browse evaluation episodes, monitor success rates, and stream 3D trajectory data seamlessly.

## Features

- **Real-time Evaluation Results Integration**:
  - Dynamically parses `eval_info.json` from `VLAMARL/eval_logs/` for multiple dataset suites (e.g., `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`).
  - Calculates and displays overall success rates, total episodes, and task-by-task success breakdowns.

- **Interactive Episode Browser & Video Player**:
  - Click on any task to view a list of all recorded episodes.
  - Automatically checks for corresponding `.mp4` video files.
  - Episodes with available videos can be clicked to open a built-in Modal Video Player, visually confirming the success or failure of the model's policy.
  - Episodes without corresponding videos are grayed out and disabled.

- **Live 3D Trajectory Visualization**:
  - Streams global states and proprioceptive data from `.parquet` files via WebSockets.
  - Real-time 3D plotting of the robot end-effector (gripper) trajectory (extracted from `observation.state`).
  - Dynamic visualizer indicating the agent's movement path using Plotly.js.

- **Asynchronous Data Flow Pipeline**:
  - Uses `asyncio` and `FastAPI` to serve the MuJoCo render loop and API endpoints concurrently.
  - Replaced thread-blocking queues with robust global frame sharing, completely eliminating "infinite loading" stalls during video streaming.

## Tech Stack
- **Backend**: Python, FastAPI, Uvicorn, PyArrow (for Parquet)
- **Frontend**: HTML5, Vanilla JavaScript, CSS3
- **Visualization**: Plotly.js (3D Trajectory), HTML5 Video Player

## Usage

### Prerequisites
Make sure you have an active conda environment with the required dependencies (e.g., `fastapi`, `uvicorn`, `pyarrow`, `libero`).

### Running the Server
The dashboard relies on a running API server that handles both the dataset parsing and the video streaming. Start the server using the following command (we recommend using `nohup` to keep it running):

```bash
conda activate libero
export MUJOCO_GL=egl
nohup python main.py > dashboard.log 2>&1 &
```

Once started, the server will host the frontend at:
`http://localhost:7052/dashboard/index.html`

### Navigating the Dashboard
1. **Dataset Selection**: On the top right, select the dataset suite you wish to inspect (e.g., `LEROBOT/10 (PARQUET)`).
2. **Evaluation Results**: The left panel will update with the success metrics.
3. **Task Breakdown**: Click on any task in the `Per-Task Breakdown` to populate the list of episodes below it.
4. **Video Playback**: Click on a highlighted episode in the `Episodes` list to pop up the video player and watch the agent's behavior for that specific evaluation run.

## Directory Structure
- `main.py`: The FastAPI server, dataset loader, and evaluation parser.
- `static/index.html`: The main dashboard layout and UI components.
- `static/dashboard/app.js`: Client-side logic, API fetching, Plotly.js trajectory rendering, and modal management.
- `static/dashboard/style.css`: Styling for the dashboard and the video modal.
