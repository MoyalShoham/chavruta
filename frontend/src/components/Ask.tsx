import { useEffect, useRef, useState, type FormEvent } from 'react'
import { api, ApiError, post } from '../api'
import { useI18n } from '../i18n'
import type { Answer, Conversation, ConversationMessage, ConversationSummary } from '../types'
import { AnswerView } from './AnswerView'

/** Questions from the dev split, which have recorded answers: they work without an API key. */
export const SAMPLES = {
  he: [
    'מאימתי קורין את שמע בערבית?',
    'מה הם ייסורים של אהבה?',
    'אני חולה ומתקשה לעמוד, האם אפשר להתפלל שמונה עשרה בישיבה?',
  ],
  en: [
    "Where in the Torah is the verse 'Hear, O Israel, the Lord is our God, the Lord is one'?",
    'Who says one must pray eighteen blessings every day, and who allows an abbreviated version?',
    'Write me a Python function that sorts a list.',
  ],
}

interface Props {
  anchorRef: string | null
  onLeaveChavruta: () => void
}

export function Ask({ anchorRef, onLeaveChavruta }: Props) {
  const { t, lang } = useI18n()
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [conversationId, setConversationId] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<ConversationSummary[]>([])
  const bottom = useRef<HTMLDivElement>(null)
  const mode = anchorRef ? 'chavruta' : 'ask'

  const loadHistory = () =>
    api<ConversationSummary[]>('/api/conversations')
      .then(setHistory)
      .catch(() => undefined)

  useEffect(() => {
    setMessages([])
    setConversationId(null)
    void loadHistory()
  }, [anchorRef])

  useEffect(() => bottom.current?.scrollIntoView?.({ block: 'end' }), [messages])

  async function send(text: string) {
    const q = text.trim()
    if (!q || busy) return
    setBusy(true)
    setError(null)
    setMessages((m) => [...m, { role: 'user', text: q, answer: null }])
    setQuestion('')
    try {
      const answer = await post<Answer>('/api/ask', {
        question: q,
        conversation_id: conversationId,
        mode,
        anchor_ref: anchorRef,
      })
      setConversationId(answer.conversation_id ?? null)
      setMessages((m) => [...m, { role: 'assistant', text: '', answer }])
      void loadHistory()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function open(id: number) {
    const c = await api<Conversation>(`/api/conversations/${id}`)
    setConversationId(c.id)
    setMessages(c.messages)
  }

  const submit = (e: FormEvent) => {
    e.preventDefault()
    void send(question)
  }

  return (
    <div className="ask-layout">
      <aside className="history">
        <button
          type="button"
          className="secondary"
          onClick={() => {
            setMessages([])
            setConversationId(null)
          }}
        >
          + {t('newConversation')}
        </button>
        <h3>{t('history')}</h3>
        <ul>
          {history.map((c) => (
            <li key={c.id}>
              <button type="button" className="link" onClick={() => void open(c.id)} dir="auto">
                {c.mode === 'chavruta' ? '📖 ' : ''}
                {c.title}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <section className="thread">
        {anchorRef && (
          <div className="chavruta-banner">
            {t('chavrutaOn')}
            <bdi>{anchorRef}</bdi>
            <button type="button" className="link" onClick={onLeaveChavruta}>
              ✕
            </button>
          </div>
        )}
        {messages.length === 0 && !anchorRef && (
          <div className="samples">
            <h3>{t('samples')}</h3>
            {SAMPLES[lang].map((s) => (
              <button
                key={s}
                type="button"
                className="sample"
                onClick={() => void send(s)}
                dir="auto"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        {messages.map((m, i) =>
          m.role === 'user' ? (
            <div key={i} className="bubble user" dir="auto">
              {m.text}
            </div>
          ) : (
            <div key={i} className="bubble assistant">
              {m.answer ? <AnswerView answer={m.answer} /> : <p dir="auto">{m.text}</p>}
            </div>
          ),
        )}
        {busy && <div className="bubble assistant thinking">{t('thinking')}</div>}
        {error && (
          <div className="notice notice-error" role="alert">
            {error}
          </div>
        )}
        <div ref={bottom} />
        <form className="ask-form" onSubmit={submit}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={t('questionPlaceholder')}
            dir="auto"
            rows={2}
            maxLength={2000}
            aria-label={t('question')}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void send(question)
              }
            }}
          />
          <button type="submit" disabled={busy || !question.trim()}>
            {t('send')}
          </button>
        </form>
      </section>
    </div>
  )
}
