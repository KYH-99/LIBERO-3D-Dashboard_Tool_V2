/* ═══════════════════════════════════════════════════════
   LIBERO_Research Dashboard — app.js
   ═══════════════════════════════════════════════════════ */

const HOST = window.location.hostname || '10.147.17.151';

const BASE = '';

// ─── DOM References ────────────────────────────────────
const successCount     = document.getElementById('success-count');
const failureCount     = document.getElementById('failure-count');
const simStatusBadge   = document.getElementById('sim-status-badge');
const taskLangBadge    = document.getElementById('task-language-badge');
const inferPointLabel  = document.getElementById('infer-point-label');
const frameLabel       = document.getElementById('frame-label');
const frameSlider      = document.getElementById('frame-slider');
const chunkProgressBar = document.getElementById('chunk-progress-bar');
const chunkProgressTxt = document.getElementById('chunk-progress-text');
const logWindow        = document.getElementById('log-window');
const actionVectorBars = document.getElementById('action-vector-bars');
const actionVectorVals = document.getElementById('action-vector-values');
const obsVectorVals    = document.getElementById('obs-vector-values');
const gripperSlider    = document.getElementById('gripper-slider');
const gripperValue     = document.getElementById('gripper-value');
const selSuite         = document.getElementById('sel-suite');
const selScenario      = document.getElementById('sel-scenario');
const selEpisode       = document.getElementById('sel-episode');
const canvas           = document.getElementById('overlay-canvas');
const ctx              = canvas.getContext('2d');
const chunkCanvas      = document.getElementById('chunk-overlay-canvas');
const chunkCtx         = chunkCanvas.getContext('2d');
const chunkRobotStatus = document.getElementById('chunk-robot-status');
const dbPreproc        = document.getElementById('data-preproc');
const dbInfer          = document.getElementById('data-infer');
const dbStep           = document.getElementById('data-step');

// ─── State ─────────────────────────────────────────────
let appState = {};
let targetChunkIndex = 0;
let chunkChart = null;
const ACTION_COLORS = ['#4f80ff','#7c4fff','#ff4f6a','#ffb547','#22d67a','#4fd1ff','#ff8c4f'];

// Local cache — fetched once per episode via REST (not WebSocket)
let localChunks = { episode_id: -1, magnitudes: [], vectors: [] };
let lastChunkProgress = -1;
let lastEpisodeId = -1;

// ─── Canvas resize ─────────────────────────────────────
function resizeCanvas() {
    canvas.width  = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    chunkCanvas.width  = chunkCanvas.parentElement.clientWidth;
    chunkCanvas.height = chunkCanvas.parentElement.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
document.getElementById('mjpeg-stream').addEventListener('load', resizeCanvas);
document.getElementById('chunk-agent-stream').addEventListener('load', () => {
    chunkCanvas.width  = chunkCanvas.parentElement.clientWidth;
    chunkCanvas.height = chunkCanvas.parentElement.clientHeight;
});
setTimeout(resizeCanvas, 800);

// ─── API Helper ─────────────────────────────────────────
function api(endpoint, body) {
    return fetch(`${BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    }).catch(err => console.error(`API ${endpoint}:`, err));
}

// ─── Chart.js Chunk Chart ──────────────────────────────
function initChunkChart() {
    const c = document.getElementById('chunkChart').getContext('2d');
    chunkChart = new Chart(c, {
        type: 'line',
        data: {
            labels: Array.from({ length: 50 }, (_, i) => i + 1),
            datasets: [{
                label: 'Action Magnitude',
                data: Array(50).fill(0),
                borderColor: '#4f80ff',
                backgroundColor: 'rgba(79,128,255,0.12)',
                borderWidth: 2,
                pointBackgroundColor: Array(50).fill('#e8eef8'),
                pointRadius: 4,
                pointHoverRadius: 7,
                fill: true,
                tension: 0.35,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { display: false },
                x: { ticks: { color: '#7a8aab', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } }
            },
            plugins: { legend: { display: false } },
            onClick: (e, elements) => {
                if (elements.length > 0) {
                    targetChunkIndex = elements[0].index + 1;
                    const n = localChunks.magnitudes.length || chunkChart.data.labels.length;
                    renderChunkChart(0, targetChunkIndex);
                    if (chunkRobotStatus)
                        chunkRobotStatus.textContent = `Selected: frames 1 → ${targetChunkIndex} / ${n}  ·  click ⚡ Run Inference to execute`;
                }
            }
        }
    });
}
initChunkChart();

// ─── WebSocket ─────────────────────────────────────────
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsHost = window.location.host || '10.147.17.151:5000';
const ws = new WebSocket(`${wsProtocol}//${wsHost}/ws/dataflow`);
ws.onmessage = e => { appState = JSON.parse(e.data); updateUI(); };
ws.onerror   = e => console.error('WS error:', e);

// ─── Fetch Chunks via REST (once per episode) ──────────
function fetchChunks(episodeId) {
    fetch(`${BASE}/api/chunks`)
        .then(r => r.json())
        .then(data => {
            localChunks = {
                episode_id: data.episode_id,
                magnitudes: data.magnitudes || [],
                vectors: data.vectors || [],
            };
            targetChunkIndex = localChunks.magnitudes.length; // default: all
            renderChunkChart(0, targetChunkIndex);
            if (chunkRobotStatus)
                chunkRobotStatus.textContent = `${localChunks.magnitudes.length} frames loaded — click a point to select target`;
        })
        .catch(console.error);
}

function renderChunkChart(prog, tgt) {
    const mags = localChunks.magnitudes;
    if (!mags.length) return;
    const n = mags.length;

    // Resize labels only if frame count changed
    if (chunkChart.data.labels.length !== n) {
        chunkChart.data.labels = Array.from({ length: n }, (_, i) => i + 1);
    }
    chunkChart.data.datasets[0].data = mags;
    chunkChart.data.datasets[0].pointBackgroundColor = mags.map((_, i) => {
        if (i < prog)  return '#22d67a';             // done
        if (i < tgt)   return '#4f80ff';             // queued
        return 'rgba(120,130,160,0.3)';              // unselected
    });
    chunkChart.data.datasets[0].pointRadius = mags.map((_, i) =>
        i === prog - 1 ? 8 : (i === tgt - 1 ? 8 : (i < tgt ? 3 : 1.5))
    );
    chunkChart.update('none');
}

// ─── Load Initial Datasets ─────────────────────────────
function loadDatasets() {
    fetch(`${BASE}/api/datasets`)
        .then(r => r.json())
        .then(data => {
            populateSelect(selSuite, data.suites || []);
            populateSelect(selScenario, data.scenarios || []);
            populateSelect(selEpisode, (data.episodes || []).map(String));
        }).catch(console.error);
}
loadDatasets();

// Refresh after state update for dynamic changes
let lastScenariosLen = 0, lastEpisodesLen = 0;

function refreshSelectsFromState() {
    if ((appState.available_scenarios || []).length !== lastScenariosLen) {
        populateSelect(selScenario, appState.available_scenarios || []);
        lastScenariosLen = (appState.available_scenarios || []).length;
    }
    if ((appState.available_episodes || []).length !== lastEpisodesLen) {
        populateSelect(selEpisode, (appState.available_episodes || []).map(String));
        lastEpisodesLen = (appState.available_episodes || []).length;
    }
}

function populateSelect(sel, items) {
    const prev = sel.value;
    sel.innerHTML = '';
    items.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item;
        opt.textContent = item;
        sel.appendChild(opt);
    });
    if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
}

// ─── Main UI Update ────────────────────────────────────
function updateUI() {
    refreshSelectsFromState();

    // Header
    successCount.textContent = appState.successes ?? 0;
    failureCount.textContent = appState.failures ?? 0;
    if (appState.sim_running) {
        simStatusBadge.textContent = '● RUNNING';
        simStatusBadge.className = 'sim-status running';
    } else {
        simStatusBadge.textContent = '● IDLE';
        simStatusBadge.className = 'sim-status';
    }
    
    // Physics Warning Banner
    const physicsBanner = document.getElementById('physics-warning-banner');
    if (physicsBanner) {
        if (appState.current_suite && appState.current_suite.toLowerCase().includes('parquet')) {
            physicsBanner.style.display = 'block';
        } else {
            physicsBanner.style.display = 'none';
        }
    }
    if (appState.current_task_language) {
        taskLangBadge.textContent = appState.current_task_language;
        const descEl = document.getElementById('dataset-task-description');
        if (descEl) descEl.textContent = appState.current_task_language;
    } else {
        taskLangBadge.textContent = '—';
        const descEl = document.getElementById('dataset-task-description');
        if (descEl) descEl.textContent = '—';
    }

    // Frame scrubber
    const cf = appState.current_frame ?? 0;
    const mf = appState.max_frames ?? 0;
    frameLabel.textContent = `Frame: ${cf} / ${mf}`;
    frameSlider.max   = mf > 0 ? mf - 1 : 0;
    frameSlider.value = cf;

    // Action vector bars
    const actionVec = appState.current_action_vector || [];
    if (actionVec.length > 0) {
        actionVectorBars.innerHTML = '';
        actionVec.forEach((val, i) => {
            const bar = document.createElement('div');
            bar.className = 'bar-item';
            const pct = Math.min(Math.abs(val) / 1.5 * 100, 100);
            bar.style.height = `${Math.max(pct, 4)}%`;
            bar.style.background = ACTION_COLORS[i % ACTION_COLORS.length];
            bar.title = `DOF ${i}: ${val}`;
            actionVectorBars.appendChild(bar);
        });
        actionVectorVals.textContent = actionVec.join(' | ');
    }

    // Obs state
    const obsVec = appState.current_obs_state || [];
    if (obsVec.length > 0) {
        obsVectorVals.textContent = obsVec.join(' | ');
    }

    // Inference point
    const pt = appState.inference_point_3d;
    if (pt && pt.length === 3) {
        inferPointLabel.textContent = `3D Inference Point: [${pt[0].toFixed(3)}, ${pt[1].toFixed(3)}, ${pt[2].toFixed(3)}]`;
    }

    // Chunk progress
    const prog = appState.chunk_progress ?? 0;
    const tgt  = appState.target_chunks  ?? 0;
    const pct  = tgt > 0 ? (prog / tgt) * 100 : 0;
    chunkProgressBar.style.width = `${pct}%`;
    chunkProgressTxt.textContent = `${prog} / ${tgt}`;

    // Chunk chart — fetch once per episode, update colors on progress change only
    const eid = appState.episode_id ?? -1;
    if (eid !== lastEpisodeId && appState.chunks_loaded) {
        lastEpisodeId = eid;
        lastChunkProgress = -1;
        fetchChunks(eid);
    } else if (prog !== lastChunkProgress) {
        lastChunkProgress = prog;
        renderChunkChart(prog, tgt);
    }

    // Pipeline stages
    document.querySelectorAll('.stage').forEach(el => el.classList.remove('active'));
    const stageId = `stage-${(appState.pipeline_stage || 'idle').toLowerCase()}`;
    const activeEl = document.getElementById(stageId);
    if (activeEl) activeEl.classList.add('active');

    // Data boxes
    const dc = appState.data_flow_content || {};
    if (appState.pipeline_stage === 'PREPROCESSING') {
        dbPreproc.textContent = JSON.stringify(dc, null, 2);
    } else if (appState.pipeline_stage === 'INFERENCE') {
        dbInfer.textContent = JSON.stringify(dc, null, 2);
    } else if (appState.pipeline_stage === 'MUJOCO_STEP') {
        dbStep.textContent = JSON.stringify(dc, null, 2);
    }

    // Logs
    const msgs = appState.log_messages || [];
    logWindow.innerHTML = '';
    msgs.slice(-20).forEach(msg => {
        const div = document.createElement('div');
        div.className = 'log-entry';
        div.textContent = msg;
        logWindow.appendChild(div);
    });
    logWindow.scrollTop = logWindow.scrollHeight;

    // Chunk robot status text
    if (chunkRobotStatus) {
        const prog = appState.chunk_progress ?? 0;
        const tgt  = appState.target_chunks  ?? 0;
        const stage = appState.pipeline_stage || 'IDLE';
        if (stage === 'MUJOCO_STEP') {
            const vec = (appState.data_flow_content || {}).action_vector || [];
            chunkRobotStatus.textContent = `Executing chunk ${prog}/${tgt} | ${vec.slice(0,4).map(v=>v.toFixed(2)).join(', ')}…`;
        } else if (stage === 'INFERENCE') {
            chunkRobotStatus.textContent = 'smolVLA inferring action chunks…';
        } else if (stage === 'PREPROCESSING') {
            chunkRobotStatus.textContent = 'Preprocessing observations…';
        } else if (stage === 'IDLE') {
            if (!chunkRobotStatus.textContent.startsWith('Selected:')) {
                chunkRobotStatus.textContent = 'Waiting for inference command';
            }
        }
    }

    drawOverlay();
    drawChunkOverlay();
}

// ─── Canvas Overlay ────────────────────────────────────
function drawOverlay() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const viewW = canvas.width / 3; // agentview is left third

    // Trajectory (red line)
    const traj = appState.trajectory || [];
    if (traj.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = '#ff4f6a';
        ctx.lineWidth = 2.5;
        ctx.shadowBlur = 6;
        ctx.shadowColor = '#ff4f6a';
        traj.forEach((pt, i) => {
            const px = (pt[0] + 0.5) * viewW;
            const py = (pt[1] + 0.5) * canvas.height;
            i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    // Inference Point (yellow dot)
    const ipt = appState.inference_point_3d;
    if (ipt && appState.pipeline_stage !== 'IDLE') {
        const px = (ipt[0] + 0.5) * viewW;
        const py = (ipt[1] + 0.5) * canvas.height;
        ctx.beginPath();
        ctx.fillStyle = '#ffb547';
        ctx.shadowBlur = 12;
        ctx.shadowColor = '#ffb547';
        ctx.arc(px, py, 9, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

// ─── Chunk Robot Canvas Overlay ────────────────────────
function drawChunkOverlay() {
    chunkCtx.clearRect(0, 0, chunkCanvas.width, chunkCanvas.height);
    const W = chunkCanvas.width;
    const H = chunkCanvas.height;

    // Trajectory (red glowing path)
    const traj = appState.trajectory || [];
    if (traj.length > 1) {
        chunkCtx.beginPath();
        chunkCtx.strokeStyle = '#ff4f6a';
        chunkCtx.lineWidth = 2;
        chunkCtx.shadowBlur = 8;
        chunkCtx.shadowColor = '#ff4f6a';
        traj.forEach((pt, i) => {
            const px = (pt[0] + 0.5) * W;
            const py = (pt[1] + 0.5) * H;
            i === 0 ? chunkCtx.moveTo(px, py) : chunkCtx.lineTo(px, py);
        });
        chunkCtx.stroke();
        chunkCtx.shadowBlur = 0;
    }

    // Inference Point (yellow pulsing dot)
    const ipt = appState.inference_point_3d;
    if (ipt && appState.pipeline_stage !== 'IDLE') {
        const px = (ipt[0] + 0.5) * W;
        const py = (ipt[1] + 0.5) * H;
        // Outer glow ring
        chunkCtx.beginPath();
        chunkCtx.strokeStyle = 'rgba(255,181,71,0.4)';
        chunkCtx.lineWidth = 4;
        chunkCtx.arc(px, py, 14, 0, Math.PI * 2);
        chunkCtx.stroke();
        // Inner dot
        chunkCtx.beginPath();
        chunkCtx.fillStyle = '#ffb547';
        chunkCtx.shadowBlur = 14;
        chunkCtx.shadowColor = '#ffb547';
        chunkCtx.arc(px, py, 7, 0, Math.PI * 2);
        chunkCtx.fill();
        chunkCtx.shadowBlur = 0;
    }

    // Chunk progress: draw a small number indicator
    const prog = appState.chunk_progress ?? 0;
    const tgt  = appState.target_chunks  ?? 50;
    if (prog > 0) {
        chunkCtx.font = 'bold 11px JetBrains Mono, monospace';
        chunkCtx.fillStyle = 'rgba(255,255,255,0.85)';
        chunkCtx.fillText(`Chunk ${prog}/${tgt}`, 8, H - 8);
    }
}

// ─── Dataset Browser Events ────────────────────────────
selSuite.addEventListener('change', () => {
    api('/api/select_suite', { suite: selSuite.value });
});
selScenario.addEventListener('change', () => {
    api('/api/select_scenario', { scenario: selScenario.value });
});
selEpisode.addEventListener('change', () => {
    api('/api/select_episode', { episode: selEpisode.value });
});

// ─── Replay Controls ───────────────────────────────────
document.getElementById('btn-play').addEventListener('click',    () => api('/api/control', { type: 'play' }));
document.getElementById('btn-pause').addEventListener('click',   () => api('/api/control', { type: 'pause' }));
document.getElementById('btn-step').addEventListener('click',    () => api('/api/control', { type: 'step_next' }));
document.getElementById('btn-reset-ep').addEventListener('click',() => api('/api/control', { type: 'sim_reset' }));

frameSlider.addEventListener('input', e => {
    const f = parseInt(e.target.value);
    frameLabel.textContent = `Frame: ${f} / ${frameSlider.max}`;
});
frameSlider.addEventListener('change', e => {
    api('/api/control', { type: 'seek', frame: parseInt(e.target.value) });
});

// ─── Sim Controls ──────────────────────────────────────
document.getElementById('btn-sim-start').addEventListener('click', () => api('/api/control', { type: 'sim_start' }));
document.getElementById('btn-sim-stop').addEventListener('click',  () => api('/api/control', { type: 'sim_stop'  }));
document.getElementById('btn-sim-reset').addEventListener('click', () => api('/api/control', { type: 'sim_reset' }));

// ─── Teleop D-Pad ──────────────────────────────────────
document.querySelectorAll('.btn-teleop').forEach(btn => {
    btn.addEventListener('click', e => {
        const payload = { type: 'teleop' };
        payload[e.target.dataset.action] = parseFloat(e.target.dataset.val);
        // Include current gripper value
        payload.gripper = parseFloat(gripperSlider.value);
        api('/api/teleop', payload);
    });
});

// ─── Gripper Slider ────────────────────────────────────
gripperSlider.addEventListener('input', e => {
    gripperValue.textContent = parseFloat(e.target.value).toFixed(2);
});
gripperSlider.addEventListener('change', e => {
    api('/api/teleop', { type: 'gripper', value: parseFloat(e.target.value) });
});

// ─── Chunk Inference ───────────────────────────────────
document.getElementById('btn-trigger-inference').addEventListener('click', () => {
    api('/api/execute_chunks', { target: targetChunkIndex });
});

// ─── Success / Failure Reporting ───────────────────────
document.getElementById('btn-report-success').addEventListener('click', () => {
    api('/api/report_success', {});
});

const failureModal  = document.getElementById('failure-modal');
document.getElementById('btn-report-failure').addEventListener('click', () => {
    failureModal.style.display = 'flex';
});
document.getElementById('btn-cancel-failure').addEventListener('click', () => {
    failureModal.style.display = 'none';
});
document.getElementById('btn-confirm-failure').addEventListener('click', () => {
    const selected = document.querySelector('input[name="failure-reason"]:checked');
    const other    = document.getElementById('failure-other-text').value.trim();
    const reason   = other || (selected ? selected.value : 'Unknown');
    api('/api/report_failure', { reason });
    failureModal.style.display = 'none';
});

// ─── Clear Logs ────────────────────────────────────────
document.getElementById('btn-clear-log').addEventListener('click', () => {
    logWindow.innerHTML = '';
    appState.log_messages = [];
});

// ─── Keyboard Shortcuts ────────────────────────────────
document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const map = {
        'ArrowLeft':  () => api('/api/teleop', { type: 'teleop', dx: -0.05, gripper: parseFloat(gripperSlider.value) }),
        'ArrowRight': () => api('/api/teleop', { type: 'teleop', dx:  0.05, gripper: parseFloat(gripperSlider.value) }),
        'ArrowUp':    () => api('/api/teleop', { type: 'teleop', dy: -0.05, gripper: parseFloat(gripperSlider.value) }),
        'ArrowDown':  () => api('/api/teleop', { type: 'teleop', dy:  0.05, gripper: parseFloat(gripperSlider.value) }),
        'w':          () => api('/api/teleop', { type: 'teleop', dz:  0.05, gripper: parseFloat(gripperSlider.value) }),
        's':          () => api('/api/teleop', { type: 'teleop', dz: -0.05, gripper: parseFloat(gripperSlider.value) }),
        ' ':          () => { e.preventDefault(); api('/api/control', { type: 'step_next' }); },
        'r':          () => api('/api/control', { type: 'sim_reset' }),
        'p':          () => api('/api/control', { type: 'play' }),
    };
    if (map[e.key]) map[e.key]();
});
