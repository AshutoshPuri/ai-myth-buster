import { useState } from 'react';

export default function ClaimInput({ onSubmit, isLoading }) {
  const [value, setValue] = useState('');

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || isLoading) return;
    onSubmit(trimmed);
    setValue('');
  };

  const handleKeyDown = (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="input-panel">
      <label className="sr-only" htmlFor="claim-input">
        Enter a claim
      </label>
      <textarea
        id="claim-input"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Enter a claim to verify..."
        rows={5}
        className="claim-input"
      />
      <div className="input-actions">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={isLoading || value.trim().length === 0}
          className="submit-button"
        >
          {isLoading ? 'Checking...' : 'Check Claim'}
        </button>
      </div>
    </div>
  );
}
