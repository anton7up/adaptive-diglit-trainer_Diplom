import { useEffect, useRef, useState } from 'react';
import api from '../api';

export default function StudentExamsPage() {
  const [exams, setExams] = useState([]);
  const [session, setSession] = useState(null);
  const [answers, setAnswers] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const taskStart = useRef({});

  useEffect(() => { api.get('/exams').then(({ data }) => setExams(data)); }, []);

  const start = async (examId) => {
    setError(''); setResult(null); setAnswers({});
    try {
      const { data } = await api.post(`/exams/${examId}/start`);
      setSession(data);
      const now = Date.now();
      taskStart.current = Object.fromEntries(data.tasks.map((task) => [task.id, now]));
    } catch (err) { setError(err?.response?.data?.detail || 'Не удалось начать экзамен'); }
  };

  const answer = async (taskId, value) => {
    if (answers[taskId]) return;
    try {
      await api.post(`/exams/sessions/${session.session_id}/answers`, {
        task_id: taskId,
        selected_option: value,
        response_time_ms: Date.now() - (taskStart.current[taskId] || Date.now()),
      });
      setAnswers((prev) => ({ ...prev, [taskId]: value }));
    } catch (err) { setError(err?.response?.data?.detail || 'Ответ не сохранён'); }
  };

  const submit = async () => {
    try {
      const { data } = await api.post(`/exams/sessions/${session.session_id}/submit`);
      setResult(data); setSession(null);
    } catch (err) { setError(err?.response?.data?.detail || 'Не удалось завершить экзамен'); }
  };

  return (
    <div>
      <header className="page-header"><div><div className="eyebrow">Контроль знаний</div><h1>Экзамены</h1></div></header>
      {error ? <div className="error-box">{error}</div> : null}
      {result ? <section className="card result-card"><h2>{result.passed ? 'Экзамен сдан' : 'Экзамен не сдан'}</h2><div className="result-score">{Math.round(result.score * 100)}%</div><p>Верных ответов: {result.correct_count} из {result.total_count}</p></section> : null}
      {!session ? <div className="exam-grid">
        {exams.map((exam) => <section className="card exam-card" key={exam.id}><h2>{exam.title}</h2><p>{exam.description}</p><div className="exam-meta"><span>{exam.duration_minutes} мин.</span><span>порог {Math.round(exam.passing_score * 100)}%</span></div><button className="primary" onClick={() => start(exam.id)}>Начать</button></section>)}
      </div> : <section className="card exam-session">
        <div className="row between"><div><div className="eyebrow">Экзамен</div><h2>{session.title}</h2></div><div className="badge">{Object.keys(answers).length}/{session.tasks.length}</div></div>
        {session.tasks.map((task) => <div className="exam-question" key={task.id}><h3>{task.position}. {task.question}</h3><div className="option-list compact">{Object.entries(task.options).map(([key, value]) => <button key={key} className={answers[task.id] === key ? 'answer-option selected' : 'answer-option'} disabled={Boolean(answers[task.id])} onClick={() => answer(task.id, key)}><span className="option-key">{key}</span><span>{value}</span></button>)}</div></div>)}
        <button className="primary" onClick={submit}>Завершить экзамен</button>
      </section>}
    </div>
  );
}
