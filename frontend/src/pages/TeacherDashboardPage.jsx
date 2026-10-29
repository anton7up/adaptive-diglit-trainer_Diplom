import { useEffect, useState } from 'react';
import api from '../api';
import MetricCard from '../components/MetricCard';

export default function TeacherDashboardPage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const load = () => api.get('/teacher/dashboard').then(({ data: payload }) => setData(payload)).catch(() => setError('Не удалось загрузить аналитику'));
  useEffect(() => {
  load();
}, []);
  return (
    <div>
      <header className="page-header"><div><div className="eyebrow">Преподаватель</div><h1>Аналитическая панель</h1></div><button className="secondary" onClick={load}>Обновить</button></header>
      {error ? <div className="error-box">{error}</div> : null}
      <div className="metrics-grid four">
        <MetricCard label="Студентов" value={data?.students_total ?? 0} />
        <MetricCard label="Активны 7 дней" value={data?.active_students_7d ?? 0} />
        <MetricCard label="Попыток" value={data?.attempts_total ?? 0} />
        <MetricCard label="Точность" value={`${Math.round((data?.overall_accuracy ?? 0) * 100)}%`} />
      </div>
      <section className="card"><h2>Компетенции</h2><div className="data-table"><div className="table-row head"><span>Навык</span><span>Попытки</span><span>Точность</span><span>Средний прогноз</span><span>Калибровочный разрыв</span></div>{(data?.skills || []).map((row) => <div className="table-row" key={row.skill_id}><span><strong>{row.name}</strong><small>{row.code}</small></span><span>{row.attempts}</span><span>{Math.round(row.accuracy * 100)}%</span><span>{row.avg_predicted_probability == null ? '—' : `${Math.round(row.avg_predicted_probability * 100)}%`}</span><span>{row.calibration_gap == null ? '—' : row.calibration_gap.toFixed(3)}</span></div>)}</div></section>
      <section className="card"><h2>Студенты</h2><div className="data-table"><div className="table-row head"><span>Студент</span><span>Попытки</span><span>Точность</span><span>Среднее время</span><span>Последняя активность</span></div>{(data?.students || []).map((row) => <div className="table-row" key={row.user_id}><span><strong>{row.full_name || row.username}</strong><small>{row.username}</small></span><span>{row.attempts}</span><span>{Math.round(row.accuracy * 100)}%</span><span>{Math.round(row.avg_response_time_ms / 1000)} c</span><span>{row.last_activity_at ? new Date(row.last_activity_at).toLocaleString() : '—'}</span></div>)}</div></section>
    </div>
  );
}
