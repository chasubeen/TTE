# Stable Test-Time Memory Expansion for Few-Shot Anomaly Detection

[DSBA 개인연구] Score-then-Expansion for Patch-based Few-Shot AD

## Table of Contents

1. [Overview](#1-overview)
2. [연구 배경 (아이디어)](#2-연구-배경-아이디어)
3. [Pipeline](#3-pipeline)
4. [핵심 발견 & 분석](#4-핵심-발견--분석)
5. [Commands](#5-commands)
6. [주요 파일들](#6-주요-파일들)
7. [Evaluation](#7-evaluation)
8. [Dependencies](#8-dependencies)

---

## 1. Overview

AnomalyDINO 스타일의 강한 patch-kNN baseline 위에서 **test-time memory expansion**을 연구하는 시스템입니다. 핵심 발견은 unlabeled test stream을 그대로 흡수하는 **naive TTE가 Ranking Collapse(정상-이상 순위 역전)로 구조적으로 붕괴**한다는 것이며, 이를 막고 coverage 이득만 취하는 **Score-then-Expansion**(흡수 전 채점 → 흡수)을 제안합니다.

### 주요 결과 (1-shot, 3-seed × 3-shuffle robust)

| Dataset | Method | I-AUROC | P-AUROC | P-AUPRO |
|---|---|:---:|:---:|:---:|
| MVTec AD | Baseline (AnomalyDINO-S) | 0.9587 | 0.9680 | 0.9188 |
|          | **Ours (Score-then-Expansion)** | **0.9647** | **0.9737** | **0.9300** |
|          | Delta | **+0.60pp** | **+0.57pp** | **+1.12pp** |
| VisA     | Baseline (AnomalyDINO-S) | 0.8377 | 0.9552 | 0.8775 |
|          | **Ours (Score-then-Expansion)** | **0.8869** | **0.9625** | **0.9070** |
|          | Delta | **+4.92pp** | **+0.73pp** | **+2.95pp** |

### 핵심 특징

- **Score-then-Expansion**: 모든 query를 흡수 *전* 메모리로 채점한 뒤 흡수 → expansion 이득이 공정하게 반영, 순위 보존
- **Ranking-Preserving Adaptation**: cross-batch context signal `g(q)`로 정상 patch 점수를 더 깎아 정상-이상 순위를 보존하는 score calibration
- **Memory-independent Gate**: M₀로만 학습한 frozen MLP(`e(q)<τ_low`) — 메모리가 커져도 흡수 기준이 표류하지 않음
- **Selective Update (reservoir)**: M₀ 영구 보존 + reservoir buffer(budget 1.5) → batch 순서에 robust
- **Backbone-agnostic Add-on**: frozen DINOv2 위에 추가 학습 없이 붙는 generic method

### 지원 데이터셋

| 데이터셋 | 카테고리 수 | 설명 |
|----------|:-----------:|------|
| MVTec AD | 15 | 산업 이상 탐지 벤치마크 |
| VisA | 12 | 복잡한 구조 / 미세 결함 탐지 (coverage gap이 큼) |

---

## 2. 연구 배경 (아이디어)

### 문제 정의: Coverage Gap

Foundation model(DINOv2) 시대에 few-shot AD의 병목은 representation이 아니라 **coverage gap**입니다. 1-shot + rotation 8× augmentation 후에도 foreground coverage는 **MVTec 17.5% / VisA 32.3%** — 정상 분포의 대부분이 memory에 표현되지 않아 FP를 유발합니다.

```
Coverage gap이란?
  Memory bank(M₀)에 저장된 정상 프로토타입이 정상 분포의 전체 영역을 커버하지 못하는 현상.
  커버되지 않는 영역의 정상 패치는 kNN distance가 높게 측정되어 anomaly로 오탐(FP)한다.
```

### 핵심 전제

> **비정상 이미지라도 대부분의 패치는 정상이다.**

이 전제를 바탕으로 unlabeled test stream에서 신뢰할 수 있는 정상 후보만 안전하게 추출하여 memory를 동적으로 보강합니다.

### 핵심 발견: 왜 naive TTE가 실패하는가

test stream의 patch를 그대로 흡수하면 오염을 막더라도 붕괴합니다. 원인은 세 가지가 결합된 **Ranking Collapse**입니다.

#### Set Monotonicity (SM)

```
SM: M' ⊇ M  →  d_k(q, M') ≤ d_k(q, M)  for all q

Memory가 커지면 정상 query든 이상 query든 kNN distance가 함께 내려간다.
→ "올바른 감소"(normal coverage 개선)와 "잘못된 감소"(anomaly가 가까워짐)를 거리만으로는 구분할 수 없다.
```

여기에 **Density Imbalance**(흡수가 특정 영역에 몰려 밀도 왜곡)와 **Moving Normality Reference**(흡수가 누적되며 정상 기준 자체가 표류)가 더해져, 정상-이상 score 순위가 역전됩니다.

### 해결: Score-then-Expansion

**핵심 아이디어**: 각 batch를 흡수 *전* 메모리로 먼저 채점하고, 정상 patch 점수를 순위 보존 방향으로 보정한 뒤 흡수하면, expansion의 이득을 붕괴 없이 취할 수 있다.

세 구성요소(Req-A/B/C)로 구현합니다.

| # | 구성요소 | 핵심 |
|---|---|---|
| ① | **Scoring** — Ranking-Preserving Adaptation (Req-C) | `s'(q) = d(q,M) · (1 − 0.5·g(q))`. `g`= 다른 이미지 FG patch와의 cosine top-k=5 평균 → 정상 patch 점수를 더 깎아 순위 보존 |
| ② | **Selection** — Memory-independent Gate (Req-B) | M₀로만 학습한 frozen MLP, `e(q)=‖MLP(q)−q‖ < τ_low`(τ_ratio 5.0)만 흡수. τ_low가 M₀에 고정 → 기준 표류 없음 |
| ③ | **Expansion** — Selective Update (Req-A) | M₀ 영구 보존 + reservoir buffer(budget 1.5 → ≤ 0.5·\|M₀\|). batch 순서에 robust |

| 속성 | naive append | Score-then-Expansion (Ours) |
|---|---|---|
| Scoring 기준 | 흡수 누적된 메모리 (표류) | 흡수 *전* 메모리 (순위 보존) |
| 흡수 기준 | 메모리 의존 (커질수록 완화) | M₀ 고정 gate (불변) |
| Memory 구조 | append-only (순서 민감) | M₀ 보존 + reservoir (순서 robust) |
| 결과 | Ranking Collapse | 전 메트릭 개선 |

### 참고 연구

| 연구 | 차용 요소 | 본 시스템 대응 |
|------|-----------|----------------|
| PatchCore (CVPR 2022) | Coreset memory bank + k-NN scoring | 기반 아키텍처 (MemoryBank) |
| AnomalyDINO (WACV 2025) | DINOv2 + rotation 8× + FG masking | Baseline |
| Online memory adaptation (FOADS 계열) | test-time memory 갱신 | TTE 동기 (naive online → Ranking Collapse 발견) |
| Manifold projector (FoundAD 계열) | normal manifold 복원 오차 | Memory-independent Gate의 MLP |

---

## 3. Pipeline

### 전체 흐름 — Score-then-Expansion (per-batch online)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PHASE                              │
├──────────────────────────────────────────────────────────────────────────┤
│  Support Image (k장) → DINOv2 ViT-S/14 (frozen) → Rotation 8× Augment     │
│                      → MemoryBank.fit   ⇒  M₀ (frozen)                    │
│                      → MLP Estimator.fit ⇒  τ_low = train_err_mean × 5.0  │
│                         (M₀로만 학습, 이후 고정 → 기준 표류 없음)           │
└──────────────────────────────────────────────────────────────────────────┘
                                     ↓
┌──────────────────────────────────────────────────────────────────────────┐
│            ONLINE TEST  —  per batch:  Score → Select → Expand            │
├──────────────────────────────────────────────────────────────────────────┤
│  현재 메모리  M = M₀ (frozen)  +  reservoir buffer                         │
│                                                                          │
│  ① SCORE   (흡수 전 채점, Req-C)                                          │
│     d(q,M) = 1-NN L2 distance                                            │
│     g(q)   = 다른 이미지 FG patch와 cosine top-k=5 평균 (cross-batch)      │
│     s'(q)  = d(q,M) · (1 − 0.5·g(q))    ← Ranking-Preserving Adaptation  │
│             + FG masking → top-1% mean  ⇒  image-level score             │
│                                                                          │
│  ② SELECT  (Memory-independent Gate, Req-B)                              │
│     e(q) = ‖MLP(q) − q‖ < τ_low   ⇒  confident-normal patches           │
│                                                                          │
│  ③ EXPAND  (Selective Update, Req-A)                                     │
│     reservoir-sample selected patches → buffer (≤ 0.5·|M₀|, budget 1.5)  │
│     M₀ 영구 보존 → 다음 batch의 M 갱신 (batch 순서 robust)                │
└──────────────────────────────────────────────────────────────────────────┘
```

### 입력 → 처리 → 출력

| 단계 | 입력 | 처리 | 출력 |
|------|------|------|------|
| Feature Extraction | 이미지 (448×448×3) | DINOv2 ViT-S/14 (frozen) | 패치 특징 (1024×384) |
| Scoring (Req-C) | 패치 특징, 현재 M | `d(q,M)·(1−0.5·g(q))` + FG masking + top-1% mean | Anomaly Score Map |
| Selection (Req-B) | 패치 특징 | MLP reconstruction error `< τ_low` | Confident-normal patches |
| Expansion (Req-A) | selected patches | reservoir sample (M₀ 보존, budget 1.5) | 갱신된 Memory Bank |

---

## 4. 핵심 발견 & 분석

### 4.1 Coverage Gap은 실재한다

15-cat MVTec AD · 12-cat VisA, 1-shot + rotation 8× 기준 foreground coverage:

| 데이터셋 | 평균 FG Coverage | 함의 |
|---|:---:|---|
| MVTec AD | 17.5% | 정상 분포의 82.5%가 M₀에 미표현 → FP 잠재 |
| VisA | 32.3% | gap은 작지만 미세 결함 → expansion 이득이 더 큼 |

### 4.2 naive TTE는 Ranking Collapse로 붕괴한다

- **Set Monotonicity**: 메모리 확장 시 정상·이상 거리가 함께 감소 → 거리만으로 "올바른 감소"와 "잘못된 감소"를 분리 불가.
- **메모리-내부 보정은 실패**: M_res·asymmetric ceiling·13종 memory-geometry/recon 신호 등 *메모리 의존* 보정은 모두 net-negative(폐기). 붕괴는 거리 분포 자체를 흔드는 구조적 문제.
- **해법의 방향**: 보정 신호는 **메모리에 독립적(exogenous)**이어야 한다 → cross-batch context signal `g(q)`.

### 4.3 Score-then-Expansion은 전 메트릭을 개선한다

7-metric, 1-shot, 3-seed × 3-shuffle robust 평균. Baseline = AnomalyDINO-S(static M₀).

**MVTec AD**

| Method | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | AUPRO |
|---|---|---|---|---|---|---|---|
| Baseline | 0.9587 | 0.9773 | 0.9596 | 0.9680 | 0.5682 | 0.5873 | 0.9188 |
| Expansion-only | 0.9622 | 0.9807 | 0.9580 | 0.9730 | 0.5780 | 0.5958 | 0.9273 |
| **+ Score Calibration** | **0.9647** | **0.9819** | **0.9598** | **0.9737** | **0.5830** | **0.5992** | **0.9300** |

**VisA**

| Method | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | AUPRO |
|---|---|---|---|---|---|---|---|
| Baseline | 0.8377 | 0.8488 | 0.8239 | 0.9552 | 0.3966 | 0.4466 | 0.8775 |
| Expansion-only | 0.8830 | 0.8857 | 0.8505 | 0.9621 | 0.4276 | 0.4750 | 0.9062 |
| **+ Score Calibration** | **0.8869** | **0.8909** | **0.8526** | **0.9625** | 0.4236 | 0.4711 | **0.9070** |

- Coverage gap이 큰 **VisA에서 이득이 더 큼** — 카테고리별 최대 **+25pp (pcb4)**, **+11pp (macaroni2)**.

### 4.4 폐기된 방향과 한계

| 시도 | 결과 | 교훈 |
|---|---|---|
| append-only + 무보정 ("frozen gate면 충분") | Era-1 stance, superseded | 흡수 누적 시 기준 표류 → 보정 필요 |
| 메모리-내부 보정 (M_res, asymmetric ceiling, 13 signals) | 모두 net-negative | 보정 신호는 메모리에 **독립적**이어야 함 |
| gap-priority append (순서 민감) | batch 순서에 취약 | reservoir random sampling으로 교체 |

> **한계 (정직한 서술)**: Score Calibration의 `g(q)`는 메모리에 독립적이라 Density Imbalance를 *예방*하지 못하고 image aggregation 단계에서 Ranking Collapse를 *가린다*(post-hoc). VisA에선 robust 양수, MVTec에선 noise 범위. 근본 해결(정상성의 **subspace** 표현으로 Set Monotonicity 탈출)은 future work.

---

## 5. Commands

> repository 루트(= `src/`를 포함한 디렉토리)에서 실행. 설정 기본값은 `src/configs/default.yaml`이며 CLI 플래그가 override.

### 기본 실행

```bash
# 현재 method (reservoir + Memory-independent gate + Ranking-Preserving Adaptation)
python -m src.main --seeds 1 2 3 --memory-policy reservoir --scoring-mode context_discount

# VisA
python -m src.main --seeds 1 2 3 --dataset VisA_pytorch \
    --memory-policy reservoir --scoring-mode context_discount

# Static AnomalyDINO baseline (두 policy 플래그 생략)
python -m src.main --seeds 1 2 3
```

### Ablation

```bash
# Budget sweep (expanded memory = M₀ × budget)
python -m src.main --memory-policy reservoir --scoring-mode context_discount --budget 1.5
python -m src.main --memory-policy reservoir --scoring-mode context_discount --budget 2.0

# Tau ratio sweep (Zone-1 흡수 임계 = train_err_mean × tau)
python -m src.main --memory-policy reservoir --scoring-mode context_discount --tau-ratio-low 3.0
python -m src.main --memory-policy reservoir --scoring-mode context_discount --tau-ratio-low 7.0

# Expansion-only (보정 끄기)
python -m src.main --memory-policy reservoir --scoring-mode baseline
```

### 특정 카테고리 / 데이터 경로 / multi-shot

```bash
# 특정 카테고리만
python -m src.main --categories bottle cable transistor \
    --memory-policy reservoir --scoring-mode context_discount

# 데이터 루트 지정 (기본값: <repo>/data)
python -m src.main --data-path /path/to/data

# k-shot
python -m src.main --shot 2 --seeds 1 2 3 --memory-policy reservoir --scoring-mode context_discount
```

- 결과: `logs/main_table_*.{json,csv}`.

---

## 6. 주요 파일들

### 핵심 컴포넌트

| 파일 | 역할 | 설명 |
|------|------|------|
| `src/main.py` | **진입점** | Score-then-Expansion 전체 파이프라인 (`python -m src.main`) |
| `src/pipeline/runner.py` | Orchestrator | training → online test 루프 + `enable_determinism` |
| `src/models/patch_extractor.py` | Feature Extractor | DINOv2 ViT-S/14 (frozen) |
| `src/memory/bank.py` · `nn.py` | Memory Bank | FAISS 기반 1-NN 프로토타입 저장소 |
| `src/memory/build.py` | M₀ 구성 | support feature → 초기 메모리 |
| `src/memory/expander.py` | **Expansion (Req-A)** | reservoir / append, M₀ 보존 (`MemoryExpander`) |
| `src/selector/mlp_gate.py` · `gate.py` | **Selection (Req-B)** | Memory-independent gate (`e<τ_low`, `Selector`) |
| `src/scorer/distance.py` | **Scoring (Req-C)** | `d(q,M)` 1-NN distance |
| `src/scorer/context.py` | Score Calibration | context signal `g` + `s'(q)=d·(1−0.5·g)` |
| `src/estimator/mlp_projector.py` | MLP Estimator | manifold projector (Zone-1 threshold) |
| `src/utils/augmentation.py` | Augmentation | Rotation 8× + PCA foreground masking |
| `src/utils/metrics.py` | Evaluation | 7 metrics (I-AUROC, P-AUROC, AUPRO 등) |
| `src/configs/default.yaml` | Configuration | 모든 하이퍼파라미터 (source of truth) |

### 디렉토리 구조

```
src/
├── main.py                   # ⭐ 진입점: Score-then-Expansion
├── pipeline/
│   ├── runner.py             # orchestrator + enable_determinism
│   ├── expansion.py          # 호환 shim → memory/
│   ├── scoring.py            # 호환 shim → scorer/
│   └── memory_build.py       # 호환 shim → memory/build.py
├── selector/                 # Req-B  Memory-independent Gate
│   ├── mlp_gate.py           #   MLP gate
│   └── gate.py               #   Selector (e < τ_low)
├── memory/                   # Req-A  Memory Bank + Expansion
│   ├── bank.py · nn.py       #   FAISS 1-NN
│   ├── build.py              #   M₀ 구성
│   └── expander.py           #   reservoir / append (MemoryExpander)
├── scorer/                   # Req-C  Scoring
│   ├── distance.py           #   d(q, M)
│   ├── context.py            #   context signal g + s'(q)=d·(1−0.5·g)
│   └── common.py
├── estimator/
│   └── mlp_projector.py      # MLP manifold projector
├── models/
│   ├── patch_extractor.py    # DINOv2 Feature Extractor
│   ├── vit_encoder.py        # ViT backbone loader
│   └── cnn_encoder.py
├── backbones/                # vendored DINOv2 (Apache-2.0)
├── data_provider/            # 데이터셋 로딩 (MVTecAD, VisA_pytorch)
├── utils/                    # augmentation · metrics · eval
└── configs/
    ├── default.yaml          # ⭐ 설정 source of truth
    └── ablation_*.yaml       # ablation presets
```

---

## 7. Evaluation

### 평가 지표 (7종)

| 레벨 | 지표 | 설명 |
|------|------|------|
| Image-level | **I-AUROC** | 이미지 단위 이상 탐지 성능 |
| Image-level | I-AP | Average Precision |
| Image-level | I-F1 | F1 Score (최적 threshold) |
| Pixel-level | **P-AUROC** | 픽셀 단위 이상 분할 성능 |
| Pixel-level | P-AP | Average Precision |
| Pixel-level | P-F1 | F1 Score |
| Pixel-level | **P-AUPRO** | Per-Region Overlap (FP-sensitive, 핵심 metric) |

### 주요 하이퍼파라미터 (`src/configs/default.yaml`)

| 파라미터 (CLI) | 기본값 | 역할 |
|---|---|---|
| `--memory-policy` | `append` (현재 method: `reservoir`) | Expansion 정책 (Req-A) |
| `--scoring-mode` | `baseline` (현재 method: `context_discount`) | Scoring/보정 (Req-C) |
| `--budget` | 1.5 | Memory 확장 배수 (expanded = M₀ × budget) |
| `--tau-ratio-low` | 5.0 | Zone-1 흡수 임계 (safe = MLP error < train_mean × tau) |
| `--shot` | 1 | Reference 이미지 수 (k-shot) |
| `--seeds` | [1] | 재현성 random seed |
| `context_lambda` / `context_k` | 0.5 / 5 | Ranking-Preserving Adaptation (S4 cross-batch) |
| `batch_size` | 8 | online batch 윈도우 |

### 재현성

- `enable_determinism(seed)`: 전체 RNG 고정 + `torch.use_deterministic_algorithms` → **동일 머신·스택에서 비트 단위 동일**.
- 정적 1-NN baseline은 머신 무관 4자리 재현. MLP-gated expansion은 GPU/cuDNN 차이로 ~1pp 변동 가능.

---

## 8. Dependencies

### 설치

```bash
conda create -n tte python=3.8 -y && conda activate tte
pip install -r requirements.txt   # DINOv2 가중치는 첫 실행 시 torch.hub로 자동 다운로드
```

### 필수 패키지 (`requirements.txt`)

```
torch>=2.0  torchvision>=0.15  numpy  scipy  scikit-learn
faiss-gpu (또는 faiss-cpu)  timm  Pillow  opencv-python  PyYAML  adeval
```

### 시스템 요구사항

| 항목 | 요구사항 |
|------|----------|
| Python | 3.8+ |
| PyTorch | 2.0+ |
| CUDA | 11.x+ (GPU 가속; faiss-gpu) |
| GPU Memory | 8GB+ 권장 |

### 데이터셋 준비

- 라이선스 제한으로 **미포함**. MVTec 형식으로 배치: `{dataset}/{category}/{train/good, test/<defect>, ground_truth/<defect>}`. 데이터 루트는 `--data-path`(기본값 `<repo>/data`).
- **MVTec AD**: https://www.mvtec.com/research-teaching/datasets/mvtec-ad → `<data>/MVTecAD/`.
- **VisA**: `VisA_20220922.tar` → 1-class split(`split_csv/1cls.csv`)으로 `<data>/VisA_pytorch/{cat}/{train/good, test/good, test/bad, ground_truth/bad}`.
- ⚠️ **VisA 마스크는 {0,255}로 이진화**(`(mask>0)*255`) — 원시 마스크는 작은 강도값이라 로더 `÷255 + >0.5`에서 사라지고 **AUPRO가 crash**.

---

## Acknowledgements

- **DINOv2** (Meta AI) — backbone via `torch.hub`; `src/backbones/dinov2/` vendored 코드는 Apache-2.0.
- **AnomalyDINO** (WACV 2025) — few-shot memory-bank baseline.
- **adeval** — anomaly detection metric 라이브러리.
- **MVTec AD**, **VisA** — 각 데이터셋 라이선스 하에 사용.

## License

미지정 (공개 전 선택, 예: MIT). vendored DINOv2는 Apache-2.0. 데이터셋·가중치는 미포함.
