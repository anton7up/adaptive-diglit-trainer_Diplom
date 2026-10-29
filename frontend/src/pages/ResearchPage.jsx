import { useEffect, useMemo, useState } from 'react';
import api from '../api';
import MetricCard from '../components/MetricCard';

function pct(value) {
  return value == null ? '—' : `${Math.round(Number(value) * 1000) / 10}%`;
}

function number(value, digits = 3) {
  return value == null || Number.isNaN(Number(value)) ? '—' : Number(value).toFixed(digits);
}

function StatusBadge({ passed }) {
  return <span className={passed ? 'badge status-pass' : 'badge status-fail'}>{passed ? 'PASS' : 'FAIL'}</span>;
}

function ResearchTable({ columns, rows, render }) {
  return (
    <div className="research-scroll">
      <table className="research-html-table">
        <thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
        <tbody>{rows.map((row, index) => <tr key={row.id || row.feature || row.group || index}>{render(row)}</tr>)}</tbody>
      </table>
    </div>
  );
}

function policyLabel(name) {
  return ({
    random: 'Случайный подбор',
    difficulty_match: 'Подбор по сложности',
    weak_skill: 'Слабый навык',
    ml_adaptive: 'ML-адаптивный',
  })[name] || name;
}

function LearningTrajectoryChart({ trajectories }) {
  if (!trajectories?.length) return null;
  const width = 820;
  const height = 280;
  const padding = { left: 52, right: 18, top: 20, bottom: 38 };
  const points = trajectories.flatMap((series) => series.points || []);
  const maxStep = Math.max(...points.map((point) => Number(point.step)), 1);
  const minY = Math.max(0, Math.min(...points.map((point) => Number(point.mean_mastery))) - 0.04);
  const maxY = Math.min(1, Math.max(...points.map((point) => Number(point.mean_mastery))) + 0.04);
  const spanY = Math.max(maxY - minY, 0.1);
  const x = (value) => padding.left + (Number(value) / maxStep) * (width - padding.left - padding.right);
  const y = (value) => padding.top + (1 - (Number(value) - minY) / spanY) * (height - padding.top - padding.bottom);
  return (
    <div className="research-chart-wrap">
      <svg className="research-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Средняя траектория мастерства по политикам">
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const value = minY + spanY * fraction;
          return <g key={fraction}><line x1={padding.left} x2={width - padding.right} y1={y(value)} y2={y(value)} className="chart-grid" /><text x={padding.left - 10} y={y(value) + 4} textAnchor="end" className="chart-label">{value.toFixed(2)}</text></g>;
        })}
        <line x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} className="chart-axis" />
        <text x={padding.left} y={height - 10} className="chart-label">0</text>
        <text x={width - padding.right} y={height - 10} textAnchor="end" className="chart-label">{maxStep} заданий</text>
        {trajectories.map((series, index) => {
          const path = (series.points || []).map((point, pointIndex) => `${pointIndex ? 'L' : 'M'} ${x(point.step)} ${y(point.mean_mastery)}`).join(' ');
          return <path key={series.policy} d={path} className={`chart-series chart-series-${index % 4}`} />;
        })}
      </svg>
      <div className="chart-legend">{trajectories.map((series, index) => <span key={series.policy}><i className={`chart-key chart-series-bg-${index % 4}`} />{policyLabel(series.policy)}</span>)}</div>
    </div>
  );
}

function BaselineBarChart({ rows }) {
  if (!rows?.length) return null;
  const max = Math.max(...rows.map((row) => Number(row.metrics?.brier || 0)), 0.001);
  return <div className="bar-chart">{rows.map((row) => <div className="bar-row" key={row.name}>
    <span>{row.name}</span>
    <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(4, Number(row.metrics?.brier || 0) / max * 100)}%` }} /></div>
    <strong>{number(row.metrics?.brier)}</strong>
  </div>)}</div>;
}

export default function ResearchPage() {
  const [card, setCard] = useState(null);
  const [quality, setQuality] = useState(null);
  const [validation, setValidation] = useState(null);
  const [profile, setProfile] = useState(null);
  const [robustness, setRobustness] = useState(null);
  const [monitoring, setMonitoring] = useState(null);
  const [cohorts, setCohorts] = useState(null);
  const [explainability, setExplainability] = useState(null);
  const [personalization, setPersonalization] = useState(null);
  const [retraining, setRetraining] = useState(null);
  const [errors, setErrors] = useState(null);
  const [curve, setCurve] = useState(null);
  const [backtest, setBacktest] = useState(null);
  const [simulation, setSimulation] = useState(null);
  const [policyStability, setPolicyStability] = useState(null);
  const [benchmark, setBenchmark] = useState(null);
  const [baselines, setBaselines] = useState(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState('');
  const [error, setError] = useState('');

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const responses = await Promise.all([
        api.get('/research/model-card'),
        api.get('/ml/quality'),
        api.get('/research/validation-protocol'),
        api.get('/research/dataset-profile'),
        api.get('/research/robustness'),
        api.get('/research/monitoring'),
        api.get('/research/cohorts?min_rows=20&min_users=2'),
        api.get('/research/explainability/global'),
        api.get('/research/personalization?max_users=18&minimum_attempts=8&top_k=5'),
        api.get('/ml/retraining-policy'),
        api.get('/research/errors'),
        api.get('/research/learning-curve'),
      ]);
      const [cardRes, qualityRes, validationRes, profileRes, robustnessRes, monitoringRes, cohortsRes, explainRes, personalizationRes, retrainRes, errorsRes, curveRes] = responses;
      setCard(cardRes.data);
      setQuality(qualityRes.data);
      setValidation(validationRes.data);
      setProfile(profileRes.data);
      setRobustness(robustnessRes.data);
      setMonitoring(monitoringRes.data);
      setCohorts(cohortsRes.data);
      setExplainability(explainRes.data);
      setPersonalization(personalizationRes.data);
      setRetraining(retrainRes.data);
      setErrors(errorsRes.data);
      setCurve(curveRes.data);
    } catch (err) {
      setError(err?.response?.data?.detail || 'Не удалось загрузить исследовательский отчёт');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const runExperiment = async (name, request, setter) => {
    setRunning(name);
    setError('');
    try {
      const response = await request();
      setter(response.data);
    } catch (err) {
      setError(err?.response?.data?.detail || `Не удалось выполнить ${name}`);
    } finally {
      setRunning('');
    }
  };

  const topFeatures = useMemo(() => (explainability?.features || []).slice(0, 12), [explainability]);
  const strongestCorrelations = profile?.strongest_univariate_correlations || [];
  const skillRobustness = robustness?.by_skill || [];

  if (loading) return <div className="loading-state">Расчёт исследовательских показателей…</div>;

  return (
    <div>
      <header className="page-header">
        <div>
          <div className="eyebrow">ВКР · ML research workspace</div>
          <h1>Диагностика, валидация и эксперименты</h1>
          <p className="muted">Основной исследовательский вопрос — сокращает ли адаптивный подбор число заданий до порога мастерства. AUC и калибровка используются как вспомогательные метрики качества прогноза.</p>
        </div>
        <button className="secondary" onClick={load}>Пересчитать отчёты</button>
      </header>

      {error ? <div className="error-box">{error}</div> : null}

      {card ? <>
        <div className="metrics-grid four">
          <MetricCard label="Экспериментов" value={card.evaluation.experiment_count ?? '—'} note="единый validation protocol" />
          <MetricCard label="ROC-AUC holdout" value={number(card.evaluation.validation?.roc_auc)} />
          <MetricCard label="PR-AUC holdout" value={number(card.evaluation.validation?.pr_auc)} />
          <MetricCard label="Brier holdout" value={number(card.evaluation.validation?.brier)} note="меньше — лучше" />
        </div>
        <section className="card">
          <div className="row between"><h2>Model Card</h2><span className="badge">{card.model_details.model_name}</span></div>
          <div className="research-grid">
            <div><span>Версия</span><strong className="mono">{card.model_details.version}</strong></div>
            <div><span>Наблюдений</span><strong>{card.training_data.rows}</strong></div>
            <div><span>Пользователей</span><strong>{card.training_data.users}</strong></div>
            <div><span>Positive rate</span><strong>{pct(card.training_data.positive_rate)}</strong></div>
            <div><span>Dataset hash</span><strong className="mono truncate">{card.model_details.dataset_hash}</strong></div>
            <div><span>ML-признаков</span><strong>{profile?.features ?? '—'}</strong></div>
          </div>
          <h3>Ограничения интерпретации</h3>
          <ul className="plain-list">{card.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
        </section>
      </> : null}

      <section className="card research-focus">
        <div className="row between"><h2>Исследовательская конструкция ВКР</h2><span className="badge">primary endpoint: mastery</span></div>
        <div className="research-grid">
          <div><span>Главные метрики</span><strong>Прирост мастерства + количество заданий до порога</strong></div>
          <div><span>Контроль</span><strong>Случайный подбор + предметные baseline</strong></div>
          <div><span>Baseline-модели</span><strong>Elo · IRT-2PL · BKT · PFA</strong></div>
        </div>
        <p className="muted">Текущая база — синтетическая. Поэтому контролируемая симуляция доказывает алгоритмическую адаптивность и воспроизводимость, но не заменяет рандомизированную проверку на реальных студентах.</p>
      </section>

      <div className="two-column">
        {validation ? <section className="card">
          <div className="row between"><h2>Validation protocol</h2><StatusBadge passed={validation.passed} /></div>
          <p className="muted">{validation.checks_passed} из {validation.checks_total} инженерных проверок пройдено.</p>
          <div className="quality-list">
            {validation.checks.map((check) => <div className="quality-row" key={check.name}>
              <span className={`status-dot ${check.passed ? 'pass' : 'fail'}`} />
              <div><strong>{check.name}</strong><small>{check.rationale}</small></div>
              <code>{check.passed ? 'PASS' : 'FAIL'}</code>
            </div>)}
          </div>
        </section> : null}

        {quality ? <section className="card">
          <div className="row between"><h2>Data Quality Gates</h2><StatusBadge passed={quality.passed} /></div>
          <div className="quality-list">
            {quality.checks.map((check) => <div className="quality-row" key={check.name}>
              <span className={`status-dot ${check.status}`} />
              <div><strong>{check.name}</strong><small>{check.message}</small></div>
              <code>{typeof check.value === 'object' ? JSON.stringify(check.value) : String(check.value)}</code>
            </div>)}
          </div>
        </section> : null}
      </div>

      {profile ? <section className="card">
        <h2>Профиль обучающего датасета</h2>
        <div className="metrics-grid four compact-metrics">
          <MetricCard label="Строк" value={profile.rows} />
          <MetricCard label="Пользователей" value={profile.user_exposure.entities} note={`Gini ${number(profile.user_exposure.gini)}`} />
          <MetricCard label="Заданий" value={profile.task_exposure.entities} note={`Gini ${number(profile.task_exposure.gini)}`} />
          <MetricCard label="Период, дней" value={number(profile.temporal.days, 1)} />
        </div>
        <h3>Сильнейшие одномерные связи с target</h3>
        <ResearchTable columns={['Признак', 'Корреляция']} rows={strongestCorrelations} render={(row) => <><td>{row.feature}</td><td>{number(row.target_correlation)}</td></>} />
        {profile.high_feature_correlations?.length ? <p className="warning-note">Обнаружены почти дублирующие признаки: {profile.high_feature_correlations.length}. Они требуют отдельной интерпретации и контроля устойчивости.</p> : <p className="success-note">Пар признаков с |r| ≥ 0.98 не обнаружено.</p>}
      </section> : null}

      <div className="two-column">
        {robustness ? <section className="card">
          <h2>Robustness analysis</h2>
          <div className="research-grid compact">
            <div><span>ROC-AUC</span><strong>{number(robustness.metrics.roc_auc)}</strong></div>
            <div><span>Brier</span><strong>{number(robustness.metrics.brier)}</strong></div>
            <div><span>F1 threshold</span><strong>{number(robustness.recommended_f1_threshold.threshold, 2)}</strong></div>
            <div><span>F1</span><strong>{number(robustness.recommended_f1_threshold.f1)}</strong></div>
          </div>
          <h3>Качество по навыкам</h3>
          <ResearchTable columns={['Навык', 'N', 'ROC-AUC', 'Brier', 'ECE']} rows={skillRobustness} render={(row) => <><td>{row.group}</td><td>{row.rows}</td><td>{number(row.roc_auc)}</td><td>{number(row.brier)}</td><td>{number(row.ece)}</td></>} />
        </section> : null}

        {personalization ? <section className="card">
          <h2>Диагностика персонализации</h2>
          <div className="research-grid compact">
            <div><span>Пользователей</span><strong>{personalization.users}</strong></div>
            <div><span>Кандидатных заданий</span><strong>{personalization.tasks}</strong></div>
            <div><span>Уникальных top-1</span><strong>{personalization.ranking_diversity.unique_first_choices}</strong></div>
            <div><span>Top-k disagreement</span><strong>{pct(personalization.ranking_diversity.pairwise_top_k_disagreement)}</strong></div>
            <div><span>Challenge-zone share</span><strong>{pct(personalization.challenge_zone.productive_zone_share)}</strong></div>
            <div><span>Ranking entropy</span><strong>{number(personalization.ranking_diversity.top_k_task_entropy)}</strong></div>
          </div>
          <p className="muted">{personalization.interpretation}</p>
        </section> : null}
      </div>

      <div className="two-column">
        {monitoring ? <section className="card">
          <div className="row between"><h2>Model monitoring</h2><StatusBadge passed={!monitoring.alerts?.length} /></div>
          <div className="research-grid compact">
            <div><span>Recent rows</span><strong>{monitoring.recent_rows}</strong></div>
            <div><span>Reference rows</span><strong>{monitoring.reference_rows}</strong></div>
            <div><span>Recent Brier</span><strong>{number(monitoring.recent_metrics?.brier)}</strong></div>
            <div><span>Recent ECE</span><strong>{number(monitoring.recent_metrics?.ece)}</strong></div>
          </div>
          <div className="feature-cloud">{monitoring.alerts?.length ? monitoring.alerts.map((alert) => <span key={JSON.stringify(alert)}>{alert.type || alert}</span>) : <span>критические сигналы отсутствуют</span>}</div>
        </section> : null}

        {retraining ? <section className="card">
          <h2>Retraining policy</h2>
          <div className="research-grid compact">
            <div><span>Решение</span><strong>{retraining.should_retrain ? 'переобучить' : 'оставить модель'}</strong></div>
            <div><span>Новых попыток</span><strong>{retraining.new_attempts_since_training}</strong></div>
            <div><span>Возраст, ч</span><strong>{retraining.hours_since_training == null ? '—' : Math.round(retraining.hours_since_training)}</strong></div>
            <div><span>Recent Brier</span><strong>{number(retraining.recent_brier)}</strong></div>
          </div>
          <div className="feature-cloud">{retraining.reasons.length ? retraining.reasons.map((reason) => <span key={reason}>{reason}</span>) : <span>триггеры отсутствуют</span>}</div>
        </section> : null}
      </div>

      <div className="two-column">
        {errors ? <section className="card">
          <h2>Error analysis</h2>
          <div className="research-grid compact">
            <div><span>Classification error</span><strong>{pct(errors.classification_error_rate)}</strong></div>
            <div><span>Mean abs. probability error</span><strong>{number(errors.mean_absolute_probability_error)}</strong></div>
            <div><span>High-confidence errors</span><strong>{errors.high_confidence_errors.length}</strong></div>
            <div><span>Skill segments</span><strong>{errors.hardest_skills.length}</strong></div>
          </div>
        </section> : null}

        {cohorts ? <section className="card">
          <h2>Cohort diagnostics</h2>
          <p className="muted">Диагностика групп используется только как описание качества модели, а не как причинный вывод.</p>
          <div className="feature-cloud">{(cohorts.cohorts || []).map((row) => <span key={row.group}>{row.group}: Brier {number(row.brier)}</span>)}</div>
        </section> : null}
      </div>

      {topFeatures.length ? <section className="card">
        <h2>Глобальная чувствительность прогноза</h2>
        <p className="muted">Контрфактическое изменение одного признака относительно опорного профиля. Это диагностическая чувствительность, а не причинная важность.</p>
        <ResearchTable columns={['Признак', 'Sensitivity', 'Low p', 'High p']} rows={topFeatures} render={(row) => <><td>{row.label || row.feature}</td><td>{number(row.sensitivity)}</td><td>{number(row.low_probability)}</td><td>{number(row.high_probability)}</td></>} />
      </section> : null}

      {curve?.rows?.length ? <section className="card">
        <h2>Learning curve</h2>
        <p className="muted">Вывод: {curve.conclusion}. Δ ROC-AUC: {number(curve.roc_auc_gain)}, улучшение Brier: {number(curve.brier_improvement)}.</p>
        <ResearchTable columns={['Доля train', 'Строк', 'ROC-AUC', 'PR-AUC', 'Brier']} rows={curve.rows} render={(row) => <><td>{Math.round(row.fraction * 100)}%</td><td>{row.training_rows}</td><td>{number(row.metrics.roc_auc)}</td><td>{number(row.metrics.pr_auc)}</td><td>{number(row.metrics.brier)}</td></>} />
      </section> : null}

      <section className="card">
        <h2>Воспроизводимые исследовательские прогоны</h2>
        <p className="muted">Тяжёлые процедуры запускаются вручную. Для защиты ключевым является experiment «адаптивность → время до мастерства», а не только качество классификатора.</p>
        <div className="experiment-actions">
          <button className="primary" disabled={!!running} onClick={() => runExperiment('policy simulation', () => api.post('/research/policy-simulation?learners=60&steps=90&mastery_threshold=0.70&minimum_skill_mastery=0.50'), setSimulation)}>{running === 'policy simulation' ? 'Adaptivity experiment…' : 'Адаптивность: порог мастерства'}</button>
          <button className="secondary" disabled={!!running} onClick={() => runExperiment('classical baselines', () => api.post('/research/classical-baselines'), setBaselines)}>{running === 'classical baselines' ? 'Baselines…' : 'Elo / IRT / BKT / PFA'}</button>
          <button className="secondary" disabled={!!running} onClick={() => runExperiment('policy robustness', () => api.post('/research/policy-robustness?learners=24&steps=90&mastery_threshold=0.70&minimum_skill_mastery=0.50'), setPolicyStability)}>{running === 'policy robustness' ? 'Robustness…' : 'Устойчивость по 5 seed'}</button>
          <button className="secondary" disabled={!!running} onClick={() => runExperiment('temporal backtest', () => api.post('/research/temporal-backtest?windows=4'), setBacktest)}>{running === 'temporal backtest' ? 'Backtest…' : 'Temporal backtest'}</button>
          <button className="secondary" disabled={!!running} onClick={() => runExperiment('model benchmark', () => api.post('/research/benchmark?full=false&folds=3'), setBenchmark)}>{running === 'model benchmark' ? 'Benchmark…' : '4-family ML benchmark'}</button>
        </div>

        {simulation ? <div className="experiment-result primary-result">
          <div className="row between"><h3>Главный результат: адаптивность</h3><StatusBadge passed={simulation.primary_endpoint?.statistically_significant} /></div>
          <div className="metrics-grid four compact-metrics">
            <MetricCard label="ML-адаптивный" value={number(simulation.primary_endpoint?.adaptive_restricted_mean_tasks, 1)} note="заданий до мастерства (restricted mean)" />
            <MetricCard label="Случайный" value={number(simulation.primary_endpoint?.random_restricted_mean_tasks, 1)} note="заданий до мастерства (restricted mean)" />
            <MetricCard label="Экономия" value={number(simulation.primary_endpoint?.tasks_saved?.estimate, 1)} note={`${pct(simulation.primary_endpoint?.relative_task_reduction)} меньше заданий`} />
            <MetricCard label="p-value" value={number(simulation.primary_endpoint?.p_value_one_sided_random_slower, 4)} note={`95% CI ${number(simulation.primary_endpoint?.tasks_saved?.lower, 1)}…${number(simulation.primary_endpoint?.tasks_saved?.upper, 1)}`} />
          </div>
          <div className="metrics-grid four compact-metrics">
            <MetricCard label="Прирост мастерства ML" value={number(simulation.learning_gain_endpoint?.adaptive_mean_mastery_gain)} />
            <MetricCard label="Прирост при random" value={number(simulation.learning_gain_endpoint?.random_mean_mastery_gain)} />
            <MetricCard label="Δ прироста" value={number(simulation.learning_gain_endpoint?.mastery_gain_advantage?.estimate)} note={`95% CI ${number(simulation.learning_gain_endpoint?.mastery_gain_advantage?.lower)}…${number(simulation.learning_gain_endpoint?.mastery_gain_advantage?.upper)}`} />
            <MetricCard label="p-value прироста" value={number(simulation.learning_gain_endpoint?.p_value_one_sided_adaptive_greater, 4)} />
          </div>
          <p className="result-claim"><strong>Формулировка для защиты:</strong> в контролируемой синтетической симуляции ML-адаптивный подбор достигает заданного порога мастерства за {number(simulation.primary_endpoint?.adaptive_restricted_mean_tasks, 1)} задания против {number(simulation.primary_endpoint?.random_restricted_mean_tasks, 1)} при случайном подборе; разница {simulation.primary_endpoint?.statistically_significant ? 'статистически значима' : 'не достигла заданного уровня значимости'} (p={number(simulation.primary_endpoint?.p_value_one_sided_random_slower, 4)}).</p>
          <p className="warning-note">Это доказательство поведения алгоритма на синтетической модели обучаемого, а не результат эксперимента на людях. Для заявления о педагогической эффективности нужен отдельный human pilot.</p>
          <LearningTrajectoryChart trajectories={simulation.learning_trajectories} />
          <ResearchTable columns={['Политика', 'Прирост мастерства', 'До порога', 'Достигли порога', 'Challenge zone']} rows={simulation.policies || []} render={(row) => <><td>{policyLabel(row.policy)}</td><td>{number(row.mastery_gain?.estimate)}</td><td>{number(row.restricted_steps_to_mastery?.estimate, 1)}</td><td>{pct(row.mastery_reach_rate)}</td><td>{pct(row.challenge_zone_share?.estimate)}</td></>} />
        </div> : null}

        {baselines ? <div className="experiment-result">
          <h3>Предметные baseline: Elo, IRT, BKT, PFA</h3>
          <p className="muted">Сравнение проводится на том же group-holdout по пользователям. Baseline подбирают параметры только по train-разделу. Brier — основная метрика вероятностного сравнения, меньше лучше.</p>
          <BaselineBarChart rows={baselines.models} />
          <ResearchTable columns={['Модель', 'ROC-AUC', 'PR-AUC', 'Brier', 'Log Loss', 'F1']} rows={baselines.models || []} render={(row) => <><td><strong>{row.name}</strong></td><td>{number(row.metrics?.roc_auc)}</td><td>{number(row.metrics?.pr_auc)}</td><td>{number(row.metrics?.brier)}</td><td>{number(row.metrics?.log_loss)}</td><td>{number(row.metrics?.f1)}</td></>} />
          <div className="feature-cloud">{(baselines.ml_vs_classical || []).map((row) => <span key={row.baseline}>ML vs {row.baseline}: ΔBrier {number(row.mean_user_brier_difference_ml_minus_baseline?.estimate)}; Holm p={number(row.p_value_holm_two_sided, 4)}{row.ml_significantly_lower_brier ? ' · ML лучше' : ' · превосходство ML не доказано'}</span>)}</div>
        </div> : null}

        {policyStability ? <div className="experiment-result">
          <h3>Устойчивость адаптивного эффекта по random seed</h3>
          <div className="research-grid compact">
            <div><span>Положительный time-to-mastery эффект</span><strong>{policyStability.summary?.positive_time_to_mastery_runs}/{policyStability.runs?.length}</strong></div>
            <div><span>Значимый time-to-mastery эффект</span><strong>{policyStability.summary?.significant_time_to_mastery_runs}/{policyStability.runs?.length}</strong></div>
            <div><span>Средняя экономия заданий</span><strong>{number(policyStability.summary?.mean_tasks_saved, 1)}</strong></div>
            <div><span>Положительный прирост mastery</span><strong>{policyStability.summary?.positive_mastery_gain_runs}/{policyStability.runs?.length}</strong></div>
          </div>
          <ResearchTable columns={['Seed', 'Экономия заданий', 'p time-to-mastery', 'Δ mastery gain', 'p mastery gain']} rows={policyStability.runs || []} render={(row) => <><td>{row.seed}</td><td>{number(row.tasks_saved, 1)}</td><td>{number(row.time_to_mastery_p_value, 4)}</td><td>{number(row.mastery_gain_advantage)}</td><td>{number(row.mastery_gain_p_value, 4)}</td></>} />
        </div> : null}

        {backtest ? <div className="experiment-result"><h3>Temporal backtest</h3><p>{backtest.windows?.length || 0} временных окон; средний ROC-AUC {number(backtest.summary?.roc_auc?.mean)}; Brier {number(backtest.summary?.brier?.mean)}.</p></div> : null}
        {benchmark ? <div className="experiment-result"><h3>ML-family benchmark</h3><p>Оценено конфигураций: {benchmark.models?.length || benchmark.results?.length || 0}. Разделение выполняется по группам пользователей.</p></div> : null}
      </section>
    </div>
  );
}
