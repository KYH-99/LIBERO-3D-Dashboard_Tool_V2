# LIBERO 3D Dashboard Tool V2

Robosuite/LIBERO 및 LeRobot (Parquet) 데이터셋을 실시간으로 모니터링하고 평가, 시각화할 수 있는 대화형 웹 대시보드입니다. 이 도구는 원시 평가 로그 파일과 직관적인 시각적 분석 사이의 간극을 메워주며, 연구자들이 모델의 평가 결과를 빠르게 탐색하고, 성공률을 추적하며, 3D 궤적 데이터를 실시간으로 확인할 수 있게 해줍니다.

## 주요 기능

- **실시간 평가 결과 통합**:
  - `VLAMARL/eval_logs/` 경로에 저장된 `eval_info.json` 파일을 동적으로 파싱하여 여러 데이터셋 환경(`libero_spatial`, `libero_object`, `libero_goal`, `libero_10` 등)의 평가 결과를 즉시 불러옵니다.
  - 전체 성공률, 총 에피소드 수, 각 태스크(Task)별 성공 내역을 정확하게 계산하여 화면에 표시합니다.

- **대화형 에피소드 브라우저 & 비디오 플레이어**:
  - 태스크(Task) 목록을 클릭하면 해당 태스크에 대한 모든 에피소드 기록이 나열됩니다.
  - 서버에 평가 영상(`.mp4`)이 존재하는지 자동으로 체크합니다.
  - 영상이 존재하는 에피소드를 클릭하면 화면 중앙에 비디오 모달(Modal) 플레이어가 팝업되어, 모델의 행동과 성공/실패 여부를 눈으로 직접 확인할 수 있습니다.
  - 영상이 없는 에피소드는 클릭할 수 없도록 반투명 처리됩니다.

- **실시간 3D 궤적(Trajectory) 시각화**:
  - WebSockets를 통해 `.parquet` 파일로부터 글로벌 상태(Global states)와 로봇의 상태 데이터를 스트리밍합니다.
  - 로봇의 상태값(`observation.state`)에서 추출한 좌표를 기반으로 로봇 팔의 그리퍼(End-effector) 궤적을 3D 그래프로 실시간으로 그려냅니다. (Plotly.js 활용)

- **비동기 데이터 흐름(Data Flow) 파이프라인**:
  - `asyncio`와 `FastAPI`를 사용하여 MuJoCo 렌더링 루프와 API 엔드포인트를 병렬로 처리합니다.
  - 영상 프레임 전송 시 발생하던 스레드 블로킹(무한 로딩 현상) 문제를 해결하기 위해 큐(Queue) 대신 전역 프레임 공유 방식을 도입하여 끊김 없는 스트리밍을 제공합니다.

## 기술 스택
- **백엔드 (Backend)**: Python, FastAPI, Uvicorn, PyArrow (Parquet 데이터 처리용)
- **프론트엔드 (Frontend)**: HTML5, Vanilla JavaScript, CSS3
- **시각화 (Visualization)**: Plotly.js (3D 궤적 렌더링), HTML5 Video Player

## 사용 방법

### 필수 구성 요소
`fastapi`, `uvicorn`, `pyarrow`, `libero` 등의 종속성이 설치된 conda 가상 환경이 활성화되어 있어야 합니다.

### 서버 실행
대시보드는 데이터셋 파싱 및 비디오 스트리밍을 처리하기 위해 API 서버를 기반으로 구동됩니다. 서버 백그라운드 유지를 위해 `nohup` 사용을 권장합니다:

```bash
conda activate libero
export MUJOCO_GL=egl
nohup python main.py > dashboard.log 2>&1 &
```

서버가 실행되면 다음 주소로 대시보드에 접속할 수 있습니다:
`http://localhost:7052/dashboard/index.html`

### 대시보드 탐색
1. **데이터셋 선택**: 우측 상단에서 분석할 데이터셋(예: `LEROBOT/10 (PARQUET)`)을 선택합니다.
2. **평가 결과 확인**: 좌측의 패널에 모델의 전체 성능 지표가 업데이트됩니다.
3. **태스크별 상세 확인**: `Per-Task Breakdown` 목록에서 특정 태스크를 클릭하여 하단의 에피소드 리스트를 불러옵니다.
4. **비디오 재생**: 하이라이트된 에피소드를 클릭하여 비디오 플레이어를 띄우고, 에이전트의 실제 평가 주행 모습을 확인합니다.

## 디렉토리 구조
- `main.py`: FastAPI 서버 구동, 데이터셋 로더, 평가 결과 파싱 핵심 로직
- `static/index.html`: 대시보드의 메인 레이아웃 및 UI 컴포넌트 뼈대
- `static/dashboard/app.js`: 클라이언트 로직, API 연동, Plotly.js 궤적 렌더링 및 모달 창 관리
- `static/dashboard/style.css`: 대시보드 및 비디오 모달용 커스텀 스타일링
