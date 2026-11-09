import { useEffect, useState, type FormEvent } from 'react'
import { api, ApiError, post } from '../api'
import { useI18n } from '../i18n'
import type { ClosedStatus, Note, User } from '../types'

export function Closed({ status }: { status: ClosedStatus }) {
  const { lang, t } = useI18n()
  const reason = lang === 'he' ? status.reason_he : status.reason
  const isShabbat = !status.reason || status.reason.startsWith('Shabbat')
  const until = status.until
    ? new Date(status.until).toLocaleString(lang === 'he' ? 'he-IL' : 'en-GB', {
        weekday: 'long',
        hour: '2-digit',
        minute: '2-digit',
      })
    : null
  return (
    <main className="closed" role="main">
      <div className="candles" aria-hidden="true">
        🕯️🕯️
      </div>
      <h1>{isShabbat ? t('closedTitle') : t('closedYomTov')}</h1>
      <p>
        {t('closedText')} {reason}.
      </p>
      {until && (
        <p className="big">
          {t('opensAt')}
          {until}
        </p>
      )}
      {status.source === 'fallback' && <p className="muted">{t('timesFallback')}</p>}
    </main>
  )
}

export function Login({ onSignedIn }: { onSignedIn: (u: User) => void }) {
  const { t } = useI18n()
  const [username, setUsername] = useState('learner')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const submit = (e: FormEvent) => {
    e.preventDefault()
    post<User>('/api/session', { username, password })
      .then(onSignedIn)
      .catch((err: unknown) => setError(err instanceof ApiError ? err.message : String(err)))
  }
  return (
    <form className="login panel" onSubmit={submit}>
      <h2>{t('signIn')}</h2>
      <label>
        {t('username')}
        <input
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
        />
      </label>
      <label>
        {t('password')}
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
        />
      </label>
      {error && (
        <div className="notice notice-error" role="alert">
          {error}
        </div>
      )}
      <button type="submit">{t('signIn')}</button>
    </form>
  )
}

export function Notes({ onOpen }: { onOpen: (ref: string) => void }) {
  const { t } = useI18n()
  const [notes, setNotes] = useState<Note[]>([])
  useEffect(() => {
    void api<Note[]>('/api/notes').then(setNotes)
  }, [])
  if (!notes.length) return <p className="muted">{t('noNotes')}</p>
  return (
    <ul className="notes-list">
      {notes.map((n) => (
        <li key={n.id} className="panel">
          <button type="button" className="link" onClick={() => onOpen(n.ref)}>
            <bdi>{n.ref}</bdi>
          </button>
          <p dir="auto">{n.body}</p>
          <span className="muted small">{new Date(n.updated_at).toLocaleDateString()}</span>
        </li>
      ))}
    </ul>
  )
}
