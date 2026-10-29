import { useEffect, useState } from 'react';
import api from '../api';

export default function TeacherExamsPage() {
  const [exams, setExams] = useState([]);
  const load = () => api.get('/exams').then(({ data }) => setExams(data));
  useEffect(() => {
  load();
}, []);
  return <div><header className="page-header"><div><div className="eyebrow">Контроль</div><h1>Экзамены</h1></div><button className="secondary" onClick={load}>Обновить</button></header><div className="exam-grid">{exams.map((exam) => <section className="card exam-card" key={exam.id}><h2>{exam.title}</h2><p>{exam.description}</p><div className="exam-meta"><span>{exam.duration_minutes} мин.</span><span>проходной {Math.round(exam.passing_score * 100)}%</span></div></section>)}</div></div>;
}
