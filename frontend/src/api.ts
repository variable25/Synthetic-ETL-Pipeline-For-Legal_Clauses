// Client for the inference API in serve/app.py. Types mirror its Pydantic models.

export const MAX_CHARS = 5000 // same limit as serve/app.py

const TIMEOUT_MS = 60_000 // a Cloud Run cold start loads the 420 MB model first

export type LabelScore = {
  label: string
  score: number
}

export type ClassifyResponse = {
  label: string // the top prediction
  scores: LabelScore[] // every label, highest first
  truncated: boolean // true if the clause was longer than 512 tokens
}

/** A failure whose message is safe to show the user as-is. */
export class ApiError extends Error {
  name = 'ApiError'
}

/** Same rules as the backend: whitespace is stripped, then 1 to MAX_CHARS characters. */
export function validateClause(text: string): string | null {
  const length = text.trim().length
  if (length === 0) return 'Paste or type a clause first.'
  if (length > MAX_CHARS) {
    return `The clause is ${length.toLocaleString('en')} characters. The limit is ${MAX_CHARS.toLocaleString('en')}.`
  }
  return null
}

export async function classify(text: string): Promise<ClassifyResponse> {
  let response: Response
  try {
    response = await fetch('/api/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'TimeoutError') {
      throw new ApiError('The server took too long to answer. Please try again.')
    }
    throw new ApiError('Could not reach the server. Check your connection and try again.')
  }

  if (response.status === 422) {
    throw new ApiError(`The clause must be 1 to ${MAX_CHARS.toLocaleString('en')} characters.`)
  }
  if (!response.ok) {
    throw new ApiError(`The server returned an error (${response.status}). Please try again.`)
  }
  return (await response.json()) as ClassifyResponse
}