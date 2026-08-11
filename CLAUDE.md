# 프로젝트 상태 (living doc — Claude가 상시 갱신)

## 과제 목표 (멘토 합의, 2026-08-04)
**새로운 데이터 증강/데이터 관련 기법으로 cross-dataset 성능(DFDC + eval2024)을 baseline 대비 향상시키기.**
- newbench(1,2)를 **학습 데이터에 포함시킨 걸 새 baseline**으로 잡는다 (eval2024·newbench 동시 향상은 어려워서, newbench는 학습에 넣고 DFDC/eval2024를 타깃으로).
- 아키텍처는 **현재 GatedDual 유지**. 성공 = 새 데이터 기법이 baseline보다 DFDC·eval2024에서 상승.
- 기법 방향은 아직 미정. (유력 후보: 압축/열화 증강 — DFDC·eval2024 둘 다 압축 영상이라 공통으로 먹힘)

## 새 baseline 정의 (`configs/gated_dual_baseline.json`, run 이름 `baseline`)
- 아키텍처: fshnt와 동일 (CLIP-L global[ln] + Swin-B/384 local[full] + FFT, gate τ2.35, SAM⊃SGD, 100ep, batch32).
- **real 풀**: FF++ youtube + **newbench1 real만** (newbench2 real은 제외 — 학습시간 억제).
- **fake 소스**: FF++ 6종(SimSwap raw, DF/F2F/FS c23 ×2, FaceShifter raw, NeuralTextures c23) + newbench1 fake + **newbench2 fake(fake_only)** + SBI.
- m = 2 + 11 fake 리스트 = **13**.
- 데이터는 전부 `data_precrop`(loose 프리크롭, 속도용) 사용.
- **평가 타깃 = DFDC + eval2024** (newbench1/2는 학습에 들어가 test 불가).

## 데이터셋 (train vs eval)
| 데이터 | 위치 | 용도 | 형태 |
|---|---|---|---|
| FF++ | `data/FaceForensics++` (+ `data_precrop`) | train | 풀프레임+lm / 프리크롭 |
| DFDC | `data/DFDC/test/frames` + labels.csv | **eval 타깃** | 256 크롭 |
| eval2024 | `eval2024/bench_crops/frames` + labels | **eval 타깃** | 256 크롭 |
| newbench1 (=nb1) | `data_precrop/newbench_1` (`train/frames` + `labels.csv`) | **train** | 프리크롭+lm |
| newbench2 (=nb2) | `data_precrop/newbench_2` (`train/frames` + `labels.csv`) | **train**(fake만) | 프리크롭+lm |

- **nb1/nb2 = newbench1/newbench2** 약칭. nb2의 동료 크롭(256 tight, landmark 없음)은 eval용이라 **학습엔 못 씀** → videos에서 풀프레임+landmark 재전처리 필요.
- nb2 videos는 real/fake 하위폴더 → `data/newbench_2/train/videos/`에 평탄 심링크 생성해둠(2400개, 라벨매칭 2362).

## 핵심 결과 (2026-08-03, PRESENT AUC) — SBI 기반
| run | 모델·데이터 | DFDC | eval2024 | newbench | newbench2 |
|---|---|---|---|---|---|
| reproduce | GatedDual·FF4 | **0.907** | 0.516 | 0.747 | (재평가 필요) |
| fshnt | GatedDual·FF6 (구 baseline) | 0.861 | 0.592 | 0.783 | (재평가 필요) |
| fshntx2 | GatedDual·FF6×2 | 0.827 | 0.584 | 0.790 | (재평가 필요) |
| dinomac | DINOMAC(DINOv3+LoRA+SBI)·FF6 | 0.820 | 0.552 | 0.745 | 0.624 |
| dinomac_nb | DINOMAC·FF6+nb1 | 0.837 | **0.626** | 0.989⚠(train-on-test) | **0.729** |

RESULTS.md/csv가 마스터 원장 (eval_all.sh / eval_dinomac.sh가 자동 갱신, merge 방식).

## 핵심 발견
- **in-the-wild 실데이터 학습 투입이 최대 레버**: nb1 투입 시 eval2024 +0.074, **독립 held-out newbench2 +0.105** (모델 무관, DINOMAC서 확인). 합성 가중치·SBI보다 큼.
- **SBI**: DFDC(실험실 swap) 크게↑, 최신 in-the-wild엔 미미/혼조.
- **branch_diag (reproduce)**: CLIP(global)이 유일 일꾼(빼면 DFDC −0.155), **Swin(86.9M=학습 99%)·FFT는 죽은 무게~배신**, 게이트는 FFT에 최고 가중치(0.47~0.52) 과신. dual+gate 구조가 값 못 함.
- **FFT는 게이트 적용됨** (멘토 확인 요청): model.py:281 `feats[i]*gth[:,i]`가 i=0,1,2 다 돌아 FFT도 게이트됨. 성능엔 도움 안 되지만 과제 담당자 요청으로 유지.

## 실행 명령 (요약)
- 학습(4-GPU): `CFG=configs/<recipe>.json bash train_ddp.sh 0,1,2,3 <run>` → `output/<run>_gated_dual`
- 평가(GatedDual): `[BATCH=64] [BENCHES=dfdc,eval2024] bash eval_all.sh 0,1,2,3 <run>`
- 평가(DINOMAC): `bash eval_dinomac.sh 0,1,2,3 <run>`
- 브랜치 진단: `python eval/branch_diag.py --run <run> --benches dfdc,eval2024,newbench2`
- 결과표: `cat output/RESULTS.md`
- 상세: `EXPERIMENTS.md`, 방법론 문서 `docs/methodology.html`, 아키텍처 `docs/gateddual_arch.html`

## 환경 제약
- **node51 = GPU 통신 불량(NCCL hang/조용한 손상)** → 학습·DDP 금지, 다른 노드 사용.
- **RTX 4090(24GB) 노드**: GatedDual 평가는 `BATCH=64` 필요(기본 256은 A6000 48GB용, 4090서 OOM).
- 학습은 A6000 권장(GatedDual full-finetune 메모리).
- 무거운 GPU 작업은 사용자가 실행(Claude 셸은 GPU 없는 로그인 노드).
