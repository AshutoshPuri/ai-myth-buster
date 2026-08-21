import { useState } from 'react'
import ClaimInput from './components/ClaimInput'
import HistoryList from './components/HistoryList'
import ResultCard from './components/ResultCard'
import { checkClaim } from './api/client'
import './App.css'

const EXAMPLE_CLAIMS = [
  'The Earth is flat.',
  'Water boils at 100 degrees Celsius at sea level.',
  'Vaccines are proven to cause autism.',
]

function App() {
  const [prediction, setPrediction] = useState(null)
  const [history, setHistory] = useState([])
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (claim) => {
    setIsLoading(true)
    setError('')

    try {
      const result = await checkClaim(claim)
      setPrediction(result)
      setHistory((prev) => [{ claim, ...result }, ...prev].slice(0, 5))
    } catch (err) {
      setError(err.message || 'Unable to classify that claim right now.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">AI myth buster</p>
          <h1>Check whether a claim is fact, myth, or half-truth.</h1>
        </div>
      </header>

      <section className="example-row" aria-label="Example claims">
        {EXAMPLE_CLAIMS.map((example) => (
          <button
            key={example}
            type="button"
            className="example-button"
            onClick={() => handleSubmit(example)}
            disabled={isLoading}
          >
            {example}
          </button>
        ))}
      </section>

      <div className="content-grid">
        <section className="panel">
          <ClaimInput onSubmit={handleSubmit} isLoading={isLoading} />
          {error && <p className="error-box">{error}</p>}
        </section>

        <aside className="panel side-panel">
          <ResultCard prediction={prediction} />
          <HistoryList entries={history} />
        </aside>
      </div>
    </main>
  )
}

export default App
