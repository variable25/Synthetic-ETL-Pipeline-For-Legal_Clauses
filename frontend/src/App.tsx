import { useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { ApiError, MAX_CHARS, classify, validateClause } from './api'
import type { ClassifyResponse } from './api'
import { EXAMPLES } from './examples'
import './App.css'

const REPO_URL = 'https://github.com/variable25/Synthetic-ETL-Pipeline-For-Legal_Clauses'
const DEFAULT_EXAMPLE = EXAMPLES.find((e) => e.label === 'Assignment') ?? EXAMPLES[0]
const MOBILE_QUERY = '(max-width: 767px)'

function formatPercent(score: number): string {
  const percent = score * 100
  return percent < 0.1 ? '<0.1%' : `${percent.toFixed(1)}%`
}

export default function App() {
  const [text, setText] = useState(DEFAULT_EXAMPLE.text)
  const [result, setResult] = useState<ClassifyResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const resultsRef = useRef<HTMLElement>(null)

  const length = text.trim().length

  async function run(clause: string) {
    if (loading) return
    const problem = validateClause(clause)
    setError(problem)
    if (problem) {
      textareaRef.current?.focus()
      return
    }

    setLoading(true)
    setResult(null)
    if (window.matchMedia(MOBILE_QUERY).matches) {
      resultsRef.current?.scrollIntoView({ block: 'start' })
    }
    try {
      setResult(await classify(clause))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void run(text)
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  function pickExample(exampleText: string) {
    setText(exampleText)
    void run(exampleText)
  }

  return (
    <>
      <header className="site-header">
        <div className="container">
          <a className="site-name" href="/">
            Clause Classifier
          </a>
          <a href={REPO_URL}>GitHub</a>
        </div>
      </header>

      <main className="container app">
        <div className="intro">
          <h1>Classify a contract clause.</h1>
          <p>
            A fine-tuned Legal-BERT model sorts a clause into one of 8 types. Nothing you paste is
            stored.
          </p>
        </div>

        <div className="workspace">
          <form className="clause-form" onSubmit={handleSubmit} noValidate>
            <label htmlFor="clause" className="field-label">
              Clause
            </label>
            <textarea
              id="clause"
              ref={textareaRef}
              value={text}
              rows={9}
              spellCheck={false}
              aria-invalid={error !== null}
              aria-describedby="clause-help clause-error"
              onChange={(event) => {
                setText(event.target.value)
                setError(null)
              }}
              onKeyDown={handleKeyDown}
            />
            <div className="form-row">
              <p id="clause-help" className={length > MAX_CHARS ? 'counter over' : 'counter'}>
                {length.toLocaleString('en')} / {MAX_CHARS.toLocaleString('en')} characters
                <span className="shortcut"> · Ctrl+Enter to classify</span>
              </p>
              <button type="submit" className="button-primary" disabled={loading}>
                {loading ? 'Classifying…' : 'Classify'}
              </button>
            </div>
            <p id="clause-error" className="field-error" role="alert">
              {error}
            </p>

            <fieldset className="examples" disabled={loading}>
              <legend>Or try an example</legend>
              <div className="example-list">
                {EXAMPLES.map((example) => (
                  <button
                    key={example.label}
                    type="button"
                    className="button-example"
                    onClick={() => pickExample(example.text)}
                  >
                    {example.label}
                  </button>
                ))}
              </div>
            </fieldset>
          </form>

          <section
            ref={resultsRef}
            className="results"
            aria-labelledby="results-title"
            aria-busy={loading}
          >
            <h2 id="results-title" className="panel-label">
              Result
            </h2>
            {loading ? (
              <Skeleton />
            ) : result ? (
              <Result result={result} />
            ) : (
              <p className="empty">
                Pick an example or paste your own clause, then press Classify. The probability of
                each of the 8 types appears here.
              </p>
            )}
            <p className="sr-only" aria-live="polite">
              {result ? `Predicted ${result.label}, ${formatPercent(result.scores[0].score)}` : ''}
            </p>
          </section>
        </div>

        <section className="metrics" aria-labelledby="metrics-title">
          <h2 id="metrics-title">Measured on real contracts</h2>
          <dl>
            <div>
              <dt>Macro-F1 on 2,424 real SEC contract clauses (LEDGAR test split)</dt>
              <dd>0.956</dd>
            </div>
            <div>
              <dt>Accuracy on the same clauses</dt>
              <dd>96.9%</dd>
            </div>
            <div>
              <dt>Zero-shot BART-MNLI baseline, macro-F1</dt>
              <dd>0.843</dd>
            </div>
            <div>
              <dt>To generate 5,064 synthetic training clauses with gpt-4o-mini</dt>
              <dd>€0.23</dd>
            </div>
          </dl>
          <p>
            The data pipeline, training and evaluation code are on <a href={REPO_URL}>GitHub</a>.
          </p>
        </section>
      </main>

      <footer className="site-footer">
        <div className="container">
          <a href="/privacy/">Privacy</a>
          <a href="/terms/">Terms</a>
          <a href={REPO_URL}>Source code</a>
        </div>
      </footer>
    </>
  )
}

function Result({ result }: { result: ClassifyResponse }) {
  return (
    <>
      <p className="top-label">{result.label}</p>
      <p className="top-score">{formatPercent(result.scores[0].score)} probability</p>
      {result.truncated && (
        <p className="note">
          This clause is longer than 512 tokens, so only its first part was classified.
        </p>
      )}
      <ol className="scores">
        {result.scores.map((s, i) => (
          <li key={s.label} className={i === 0 ? 'score top' : 'score'}>
            <span className="score-label">{s.label}</span>
            <span className="score-bar" aria-hidden="true">
              <span style={{ width: `${s.score * 100}%` }} />
            </span>
            <span className="score-value">{formatPercent(s.score)}</span>
          </li>
        ))}
      </ol>
    </>
  )
}

function Skeleton() {
  return (
    <div className="skeleton">
      <p className="note">Running the model. The first request can take up to 30 seconds.</p>
      <div aria-hidden="true">
        <span className="skeleton-title" />
        {EXAMPLES.map((e) => (
          <span key={e.label} className="skeleton-row" />
        ))}
      </div>
    </div>
  )
}