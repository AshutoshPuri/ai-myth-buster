const LABEL_COLORS = {
  Fact: '#16a34a',
  Myth: '#dc2626',
  'Half-Truth': '#d97706',
};

export default function HistoryList({ entries }) {
  if (!entries.length) {
    return (
      <div className="history-panel empty">
        No recent checks yet.
      </div>
    );
  }

  return (
    <div className="history-panel">
      <h3>Recent checks</h3>
      <ul className="history-list">
        {entries.map((entry, index) => (
          <li key={`${entry.claim}-${index}`} className="history-item">
            <div className="history-main">
              <span className="history-claim">{entry.claim}</span>
              <span
                className="history-badge"
                style={{ backgroundColor: LABEL_COLORS[entry.label] || '#6b7280' }}
              >
                {entry.label}
              </span>
            </div>
            <small>{Math.round(entry.confidence * 100)}% confidence</small>
          </li>
        ))}
      </ul>
    </div>
  );
}
