# v3.3-shadow backtest bundle

모바일 사용자를 위한 GitHub Actions 실행판입니다.

- cutoff 이전 데이터만으로 E/S/R/G/안정성/T-score/상태전이를 계산합니다.
- `frozen_predictions.csv`는 미래 결과를 제외한 동결 판정입니다.
- `outcomes.csv`는 이후 180일 최대배수와 2x/3x/5x/10x 여부를 붙입니다.
- `summary.csv`는 클래스별 성과와 Wilson 95% 신뢰구간을 냅니다.
- 시장 레짐 게이트는 아직 v3.4 후보라 포함하지 않았습니다.

## v3.3-shadow 구현 1.0에서 고정한 모호성
1. ATR14 = Wilder 방식.
2. R의 H = +50% 도달 후 running high를 추적하고, 종가가 running high 대비 -10% 이하인 상태가 5거래일 연속 시작되는 첫 구간을 pullback 시작으로 정의.
3. E 저장 확인 = 7일 storage 양수 + 이벤트 후 7일 저점이 이벤트 전 20일 핵심 저점 미이탈 + 이벤트 후 7일 종가 중앙값이 이벤트 전 20일 종가 중앙값보다 높음.
4. S→E = 최근 60일 VR>=2 이벤트에서 상승봉+상단절반 마감 또는 3일 내 +8% 반응.
5. 제자리걸음 패널티 = 최근60일 VR>=2 이벤트 2회 이상 + 최근20/직전20 종가 중앙값 변화 ±5% + 저점 기준 상승 +2% 이하.
6. S가 최강 엔진인데 S→E가 없으면 FIRE 승격 금지.

## 모바일 실행
GitHub → Actions → `v3.3 shadow backtest` → Run workflow → cutoff 입력 → Run.
