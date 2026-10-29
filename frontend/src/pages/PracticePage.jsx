import { useEffect, useRef, useState } from 'react';
import api from '../api';

export default function PracticePage() {
  const [skills, setSkills] = useState([]);
  const [skillId, setSkillId] = useState('');
  const [task, setTask] = useState(null);
  const [answer, setAnswer] = useState('');
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const startedAt = useRef(Date.now());

  useEffect(() => {
    api.get('/skills').then(({ data }) => setSkills(data)).catch(() => setError('Не удалось загрузить компетенции'));
  }, []);

  const loadTask = async () => {
    setLoading(true); setError(''); setAnswer(''); setFeedback(null);
    try {
      const { data } = await api.post('/practice/next', { skill_id: skillId ? Number(skillId) : null });
      setTask(data); startedAt.current = Date.now();
    } catch (err) {
      setTask(null); setError(err?.response?.data?.detail || 'Не удалось получить задание');
    } finally { setLoading(false); }
  };

  useEffect(() => { loadTask(); }, [skillId]);

  const submit = async () => {
    if (!task || !answer) return;
    const elapsed = Math.max(0, Date.now() - startedAt.current);
    try {
      const { data } = await api.post('/practice/submit', {
        task_id: task.id,
        selected_option: answer,
        response_time_ms: elapsed,
        hint_used: false,
        session_id: 'web-practice',
      });
      setFeedback(data);
    } catch (err) {
      setError(err?.response?.data?.detail || 'Не удалось отправить ответ');
    }
  };

  return (
    <div>
      <header className="page-header">
        <div><div className="eyebrow">Адаптивный режим</div><h1>Практика</h1></div>
        <label className="inline-select">Компетенция
          <select value={skillId} onChange={(e) => setSkillId(e.target.value)}>
            <option value="">Смешанная практика</option>
            {skills.map((skill) => <option key={skill.id} value={skill.id}>{skill.name}</option>)}
          </select>
        </label>
      </header>

      {error ? <div className="error-box">{error}</div> : null}
      <section className="card task-card">
        {loading ? <div className="loading-state">ML-модуль подбирает задание…</div> : null}
        {!loading && task ? <>
          <div className="task-meta">
            <span className="badge">{task.skill_name}</span>
            <span>сложность {task.difficulty.toFixed(2)}</span>
            <span>уровень {task.cognitive_level}</span>
            <span>≈ {task.estimated_time_sec} сек.</span>
          </div>
          <h2>{task.question}</h2>
          <div className="option-list">
            {Object.entries(task.options).map(([key, value]) => (
              <button key={key} className={answer === key ? 'answer-option selected' : 'answer-option'} onClick={() => !feedback && setAnswer(key)}>
                <span className="option-key">{key}</span><span>{value}</span>
              </button>
            ))}
          </div>
          <div className="task-actions">
            <button className="primary" onClick={submit} disabled={!answer || Boolean(feedback)}>Проверить</button>
            <button className="secondary" onClick={loadTask}>Следующее задание</button>
          </div>
          <div className="prediction-strip">
            <div><span>Прогноз успеха</span><strong>{Math.round(task.predicted_probability * 100)}%</strong></div>
            <div><span>ML-версия</span><strong>{task.model_version}</strong></div>
            <div><span>Adaptive score</span><strong>{task.selection_score.toFixed(3)}</strong></div>
          </div>
        </> : null}
      </section>

      {feedback ? <section className={feedback.is_correct ? 'card feedback-card success' : 'card feedback-card fail'}>
        <h3>{feedback.is_correct ? 'Ответ верный' : 'Ответ неверный'}</h3>
        <p>{feedback.explanation}</p>
        <div>Правильный вариант: <strong>{feedback.correct_option}</strong></div>
        {feedback.predicted_probability !== null ? <div>Прогноз перед ответом: <strong>{Math.round(feedback.predicted_probability * 100)}%</strong></div> : null}
      </section> : null}
    </div>
  );
}
