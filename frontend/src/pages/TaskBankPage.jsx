import { useEffect, useState } from 'react';
import api from '../api';

export default function TaskBankPage() {
  const [skills, setSkills] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [skillId, setSkillId] = useState('');
  const load = async () => {
    const [s, t] = await Promise.all([api.get('/skills'), api.get('/tasks', { params: skillId ? { skill_id: Number(skillId) } : {} })]);
    setSkills(s.data); setTasks(t.data);
  };
  useEffect(() => { load(); }, [skillId]);
  return (
    <div>
      <header className="page-header"><div><div className="eyebrow">Контент</div><h1>Банк заданий</h1></div><label className="inline-select">Компетенция<select value={skillId} onChange={(e) => setSkillId(e.target.value)}><option value="">Все</option>{skills.map((skill) => <option key={skill.id} value={skill.id}>{skill.name}</option>)}</select></label></header>
      <section className="card"><div className="data-table tasks-table"><div className="table-row head"><span>ID / вопрос</span><span>Сложность</span><span>Уровень</span><span>Время</span><span>Статус</span></div>{tasks.map((task) => <div className="table-row" key={task.id}><span><strong>#{task.id} {task.question}</strong><small>{task.tags.join(', ')}</small></span><span>{task.difficulty.toFixed(2)}</span><span>{task.cognitive_level}</span><span>{task.estimated_time_sec} c</span><span>{task.active ? 'активно' : 'выкл.'}</span></div>)}</div></section>
    </div>
  );
}
