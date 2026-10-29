import { useEffect, useState } from 'react';
import api from '../api';
import MetricCard from '../components/MetricCard';

function metric(value) { return value == null ? '—' : Number(value).toFixed(3); }

export default function MLPage({ user }) {
  const [status, setStatus] = useState(null);
  const [models, setModels] = useState([]);
  const [training, setTraining] = useState(false);
  const [bootstrap, setBootstrap] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const load = async () => {
    try {
      const [statusResponse, modelsResponse] = await Promise.all([api.get('/ml/status'), api.get('/ml/models')]);
      setStatus(statusResponse.data); setModels(modelsResponse.data);
    } catch (err) { setError(err?.response?.data?.detail || 'Не удалось загрузить ML-статус'); }
  };
  useEffect(() => { load(); }, []);

  const runTraining = async () => {
    setTraining(true); setError(''); setMessage('');
    try {
      const { data } = await api.post('/ml/train');
      setMessage(`Обучение завершено: ${data.model_name}, ${data.dataset_rows} наблюдений.`);
      await load();
    } catch (err) { setError(err?.response?.data?.detail || 'Ошибка обучения'); }
    finally { setTraining(false); }
  };

  const generateBootstrap = async () => {
    setBootstrap(true); setError(''); setMessage('');
    try {
      const { data } = await api.post('/ml/bootstrap', { users: 36, attempts_per_user: 55 });
      setMessage(`Создано синтетических попыток: ${data.attempts_created}. Они помечены как bootstrap и не смешиваются с активной учебной статистикой.`);
    } catch (err) { setError(err?.response?.data?.detail || 'Ошибка bootstrap'); }
    finally { setBootstrap(false); }
  };

  const activeMetrics = status?.metrics?.validation || status?.metrics || {};
  return (
    <div>
      <header className="page-header"><div><div className="eyebrow">Machine Learning</div><h1>Модельный контур</h1></div><div className="row gap"><button className="secondary" onClick={load}>Обновить</button><button className="primary" onClick={runTraining} disabled={training}>{training ? 'Обучение…' : 'Переобучить модели'}</button></div></header>
      {error ? <div className="error-box">{error}</div> : null}{message ? <div className="success-box">{message}</div> : null}
      <div className="metrics-grid four"><MetricCard label="Активная модель" value={status?.model_name || 'нет'} note={status?.version} /><MetricCard label="ROC-AUC" value={metric(activeMetrics.roc_auc)} /><MetricCard label="Brier score" value={metric(activeMetrics.brier)} /><MetricCard label="Log-loss" value={metric(activeMetrics.log_loss)} /></div>
      <section className="card"><div className="row between"><div><h2>ML pipeline</h2><p className="muted">{status?.feature_names?.length || 0} признаков → feature engineering по истории попыток → group-aware validation → 12 конфигураций четырёх семейств → выбор по ROC-AUC/PR-AUC/F1/LogLoss/Brier → versioned artifact → online inference.</p></div>{user.role === 'admin' ? <button className="secondary" onClick={generateBootstrap} disabled={bootstrap}>{bootstrap ? 'Генерация…' : 'Создать demo bootstrap'}</button> : null}</div><div className="feature-cloud">{(status?.feature_names || []).map((feature) => <span key={feature}>{feature}</span>)}</div></section>
      <section className="card"><h2>Реестр моделей</h2><div className="data-table"><div className="table-row head"><span>Версия</span><span>Алгоритм</span><span>Статус</span><span>ROC-AUC</span><span>Brier</span></div>{models.map((model) => <div className="table-row" key={model.version}><span><strong>{model.version}</strong><small>{new Date(model.trained_at).toLocaleString()}</small></span><span>{model.model_name}</span><span>{model.is_active ? <span className="badge">ACTIVE</span> : 'архив'}</span><span>{metric(model.metrics?.validation?.roc_auc)}</span><span>{metric(model.metrics?.validation?.brier)}</span></div>)}</div></section>
    </div>
  );
}
