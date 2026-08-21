const LABEL_COLORS = {
  Fact: '#16a34a',
  Myth: '#dc2626',
  'Half-Truth': '#d97706',
};

export default function ResultCard({ prediction }) {
  if (!prediction) return null;

  const { label, confidence, probabilities } = prediction;
  const color = LABEL_COLORS[label] || '#6b7280';
  const classEntries = Object.entries(probabilities || {});

  return (
    <div className="result-card">
      <div className="result-header">
        <span className="result-label" style={{ backgroundColor: color }}>
          {label}
        </span>
        <span className="confidence">{Math.round(confidence * 100)}%</span>
      </div>

      <div className="probability-list">
        {classEntries.map(([name, value]) => (
          <div key={name} className="probability-row">
            <div className="probability-meta">
              <span>{name}</span>
              <span>{Math.round(value * 100)}%</span>
            </div>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: `${Math.max(4, value * 100)}%`,
                  backgroundColor: LABEL_COLORS[name] || '#6b7280',
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
