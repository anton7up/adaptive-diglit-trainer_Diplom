# ML architecture

## Цель

ML-подсистема оценивает вероятность правильного ответа конкретного пользователя на конкретное задание и передаёт эту вероятность policy-слою адаптивного ранжирования.

## Feature engineering

Обучающая строка строится для момента **до текущей попытки**. Текущий target не участвует в вычислении признаков этой же строки. Это принципиально для предотвращения прямой утечки целевой переменной.

Группы признаков:

1. общая история пользователя;
2. rolling accuracy и activity windows;
3. история пользователя по навыку;
4. история конкретного задания;
5. временные интервалы между попытками;
6. response-time statistics;
7. streak features;
8. характеристики задания;
9. difficulty/speed interaction features;
10. activity density и cyclic time features;
11. cold-start flags;
12. categorical skill code.

Всего в текущей схеме — 39 признаков.

## Экспериментальное пространство

Обучаются 8 конфигураций из четырёх семейств:

- Logistic Regression — 2 варианта регуляризации;
- Random Forest — 2 варианта;
- Extra Trees — 2 варианта;
- HistGradientBoosting — 2 варианта.

Набор ограничен двумя содержательно различающимися конфигурациями каждого семейства, чтобы полный воспроизводимый цикл обучения оставался вычислительно устойчивым на расширенном датасете.

Разделение выполняется по пользователям. Пользователь не должен одновременно попадать в train и validation holdout.

## Метрики

Для каждой модели рассчитываются:

- accuracy;
- precision;
- recall;
- F1;
- ROC-AUC;
- PR-AUC;
- log loss;
- Brier score;
- calibration diagnostics.

Победитель выбирается по составной функции, а не по одной accuracy.

Отдельно реализован предметный baseline benchmark: Elo, online IRT-2PL, BKT и PFA. Он использует тот же group-aware holdout по обучаемым. Baseline-параметры выбираются только на train-пользователях, а paired Brier-сравнения агрегируются на уровне пользователя.

## Model registry

Выбранная модель переобучается, сериализуется в joblib и получает уникальную версию. Metadata фиксирует набор признаков, хэш датасета, параметры, split и результаты сравнения кандидатов.

## Adaptive policy

Inference-модель возвращает `P(correct)`. Затем policy-слой отдельно рассчитывает utility задания. Это разделение позволяет не смешивать статистический прогноз и педагогическое решение.

Основная исследовательская метрика policy-слоя — **time-to-mastery**, а не ROC-AUC модели. В воспроизводимой симуляции сравниваются random, difficulty-match, weak-skill и ML-adaptive policies. Для matched learners фиксируются одинаковые начальные профили и общие random streams. Рассчитываются restricted mean tasks-to-mastery, bootstrap CI, paired randomization p-value, effect size, mastery reach rate и траектории мастерства.

При ранжировании учитываются:

- расстояние до целевой productive learning zone;
- дефицит навыка;
- uncertainty;
- novelty;
- соответствие когнитивного уровня;
- штраф за недавнее повторение.

## Monitoring

Подсистема включает data-quality gates, calibration diagnostics, drift indicators, cohort diagnostics, temporal analysis и retraining policy. Это позволяет отделить обучение модели от её последующей эксплуатации.

## External validity

Поставляемый bootstrap interaction log синтетический. Поэтому результаты policy simulation являются алгоритмической валидацией. Вывод о педагогической эффективности требует отдельного randomized human pilot; его дизайн зафиксирован в `docs/HUMAN_VALIDATION_PROTOCOL.md`.
