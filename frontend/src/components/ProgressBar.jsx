export default function ProgressBar({ value, max = 1, label }) {
  const percent = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="progress-block">
      <div className="progress-caption">
        <span>{label}</span>
        <strong>{Math.round(percent)}%</strong>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}
