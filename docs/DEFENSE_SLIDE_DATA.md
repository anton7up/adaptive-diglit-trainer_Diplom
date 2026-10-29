# Данные для ключевого слайда защиты

## Заголовок

**Адаптивный подбор доводит до порога мастерства за меньшее число заданий**

## Ключевой результат

- ML-адаптивный подбор: **73,3 задания**
- случайный подбор: **84,6 задания**
- экономия: **11,3 задания (13,4%)**
- 95% CI: **8,2–14,6**
- **p < 0,001**, Cohen's dz = **0.90**
- mastery reached within horizon: **70% vs 25%**

Подпись: matched synthetic simulation, **60 обучаемых × 90 заданий**, mastery criterion: mean latent mastery ≥ 0,70 и каждый skill ≥ 0,50.

## Второй исследовательский результат

Средний прирост мастерства: **0.211 vs 0.162**, преимущество **0.048**, 95% CI **0.044–0.053**, p < 0,001.

## Устойчивость

Пять независимых seed: положительный и статистически значимый time-to-mastery эффект в **5/5** прогонов; средняя экономия **8,1 задания**.

## Baseline

| Модель | Brier ↓ |
|---|---:|
| ML | 0.1891 |
| IRT-2PL | 0.2016 |
| Elo | 0.2019 |
| PFA | 0.2184 |
| BKT | 0.2483 |

На текущем learner-disjoint synthetic holdout ML имеет наименьший Brier; paired user-level сравнения с Holm-коррекцией значимы для всех четырёх baseline.

## Методологическая оговорка

Данные и policy experiment синтетические. Результат подтверждает алгоритмическую адаптивность; внешняя педагогическая эффективность должна проверяться на реальной выборке по подготовленному randomized pilot protocol.

## Графики

- `docs/figures/time_to_mastery.svg` — ключевое сравнение adaptive vs random;
- `docs/figures/policy_time_to_mastery.svg` — четыре политики;
- `docs/figures/mastery_reach_rate.svg` — достижение порога;
- `docs/figures/mastery_trajectories.svg` — динамика мастерства;
- `docs/figures/baseline_brier.svg` — ML против Elo / IRT / BKT / PFA.
