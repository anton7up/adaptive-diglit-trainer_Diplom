import { useEffect, useState } from 'react';
import api from '../api';
import MetricCard from '../components/MetricCard';
import ProgressBar from '../components/ProgressBar';

export default function ProgressPage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const load = () => api.get('/progress/me').then(({ data: payload }) => setData(payload)).catch(() => setError('Не удалось загрузить прогресс'));
  useEffect(() => {
  load();
}, []);
  return (
    <div>
      <header className="page-header"><div><div className="eyebrow">Персональная аналитика</div><h1>Прогресс</h1></div><button className="secondary" onClick={load}>Обновить</button></header>
      {error ? <div className="error-box">{error}</div> : null}
      <div className="metrics-grid">
        <MetricCard label="Попыток" value={data?.attempts_total ?? 0} />
        <MetricCard label="Верных ответов" value={data?.correct_total ?? 0} />
        <MetricCard label="Общая точность" value={`${Math.round((data?.overall_accuracy ?? 0) * 100)}%`} />
      </div>
      <section className="card">
        <h2>Компетенции</h2>
        <div className="skills-grid">
          {(data?.rows || []).map((row) => (
            <div className="skill-card" key={row.skill_id}>
              <div className="skill-title"><strong>{row.skill_name}</strong><span>{row.attempts_count} попыток</span></div>
              <ProgressBar label="Оценка освоения" value={row.mastery_estimate} />
              <div className="skill-stats">
                <span>Точность <strong>{Math.round(row.accuracy * 100)}%</strong></span>
                <span>Последние 5 <strong>{Math.round(row.rolling_accuracy_5 * 100)}%</strong></span>
                <span>Среднее время <strong>{Math.round(row.mean_response_time_ms / 1000)} c</strong></span>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
