# Stable Test-Time Memory Expansion for Few-shot Anomaly Detection

Few-shot 이상 탐지에서 test-time memory expansion이 일으키는 **Ranking Collapse**를 막고, coverage 이득만 안전하게 취하는 **Score-then-Expansion** 파이프라인.

## 개요

- **배경**: memory-bank AD(PatchCore / AnomalyDINO)는 소수 정상 이미지로 M₀를 만들고 1-NN 거리로 이상을 판정 → few-shot에선 정상 변동을 못 담아 **Coverage Gap**(MVTec 17.5% / VisA 32.3%) 발생 → 오탐.
- **시도**: 테스트 스트림의 정상 patch를 메모리에 흡수(**TTE**)하면 gap을 메울 수 있음.
- **문제**: naive TTE는 오염을 막아도 붕괴 — patch 추가 시 모든 query 거리 감소(**Set Monotonicity**) + **Density Imbalance** · **Moving Normality Reference** → 정상-이상 순위 역전(**Ranking Collapse**).
- **제안**: 붕괴 없이 흡수하는 3-구성요소 **Score-then-Expansion**(흡수 전 채점 → 흡수).

## 방법 (Score-then-Expansion)

| # | 구성요소 | 핵심 |
|---|---|---|
| ① | **Scoring** — Ranking-Preserving Adaptation | `s'(q) = d(q,M)·(1 − λ·g(q))`. `g`=다른 이미지 FG patch와의 cosine top-k=5 평균 → 정상 patch 점수를 더 깎아 순위 보존. |
| ② | **Selection** — Memory-independent Gate | M₀로만 학습한 frozen MLP, `e(q)=‖MLP(q)−q‖ < τ_low`만 흡수. τ_low가 M₀에 고정 → 기준 표류 없음. |
| ③ | **Expansion** — Selective Update | M₀ 영구 보존 + reservoir buffer(budget 1.5 → ≤0.5·\|M₀\|). batch 순서에 robust. |

- Backbone: frozen **DINOv2 ViT-S/14** (`torch.hub`), 1-shot + rotation 8×.
- 한계: score calibration은 사후 보정(post-hoc) — VisA는 robust 양수, MVTec은 noise 범위. 근본 해결(정상성의 **subspace** 표현)은 future work.

## 결과

1-shot, 3-seed × 3-shuffle robust. Baseline = AnomalyDINO-S(static M₀).

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

- Coverage Gap이 큰 **VisA에서 이득이 더 큼** (cat별 최대 +25pp pcb4 / +11pp macaroni2).

## 구조

```
src/                python -m src.main
  selector/         Req-B  MLP gate (mlp_gate.py) + Selector(e<τ_low)
  memory/           Req-A  bank.py·nn.py (FAISS) + expander.py (reservoir/append) + build.py (M₀)
  scorer/           Req-C  distance.py (d(q,M)) + context.py (g + discount) + common.py
  pipeline/         runner.py  (expansion/scoring/memory_build = 호환 shim)
  models/ backbones/  DINOv2 ViT-S/14 extractor (backbones/dinov2 vendored)
  data_provider/ configs/ utils/ estimator/ main.py
requirements.txt  README.md  .gitignore
```

## 설치

```bash
conda create -n tte python=3.8 -y && conda activate tte
pip install -r requirements.txt   # DINOv2 가중치는 첫 실행 시 torch.hub로 자동 다운로드
```

## 데이터

- 라이선스 제한으로 미포함. MVTec 형식으로 배치: `{dataset}/{category}/{train/good, test/<defect>, ground_truth/<defect>}`.
- **MVTec AD**: https://www.mvtec.com/research-teaching/datasets/mvtec-ad → `<data>/MVTecAD/`.
- **VisA**: `VisA_20220922.tar` → 1-class split(`split_csv/1cls.csv`)로 `<data>/VisA_pytorch/{cat}/{train/good, test/good, test/bad, ground_truth/bad}`.
- ⚠️ **VisA 마스크는 {0,255}로 이진화**(`(mask>0)*255`) — 원시 마스크는 작은 강도값이라 로더 `÷255 + >0.5`에서 사라지고 AUPRO가 crash.
- 데이터 루트는 `--data-path`로 지정(기본값 `<repo>/data`).

## 사용법

```bash
# 현재 method (reservoir + Memory-independent gate + Ranking-Preserving Adaptation)
python -m src.main --dataset MVTecAD --seeds 1 2 3 \
    --memory-policy reservoir --scoring-mode context_discount --data-path /path/to/data

# Static baseline (두 플래그 생략)
python -m src.main --dataset MVTecAD --seeds 1 2 3 --data-path /path/to/data
```

- 결과: `logs/main_table_*.{json,csv}`.
- 하이퍼파라미터(`budget`, `tau_ratio_low`, `context_signal/lambda/k`)는 `src/configs/default.yaml`, CLI override 가능.

## 재현성

- `enable_determinism(seed)`: 전체 RNG 고정 + `torch.use_deterministic_algorithms` → **동일 머신·스택에서 비트 단위 동일**.
- 표는 3 seed × 3 shuffle robust 평균. `--seeds 1 2 3`로 multi-seed 평균.
- 정적 1-NN baseline은 머신 무관 4자리 재현. MLP-gated expansion은 GPU/cuDNN 차이로 ~1pp 변동 가능.

## Acknowledgements

- **DINOv2** (Meta AI) — backbone via `torch.hub`; `src/backbones/dinov2/` vendored 코드는 Apache-2.0.
- **AnomalyDINO** (WACV 2025) — few-shot memory-bank baseline.
- **adeval** — metric 라이브러리.
- **MVTec AD**, **VisA** — 각 데이터셋 라이선스 하에 사용.

## License

미지정 (공개 전 선택, 예: MIT). vendored DINOv2는 Apache-2.0. 데이터셋·가중치는 미포함.
