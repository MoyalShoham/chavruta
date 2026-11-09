import { useEffect, useState } from 'react'
import { api, ApiError, del, post, put, query } from '../api'
import { dirOf, useI18n } from '../i18n'
import type { Card, Links, Note, Sugya, WorkInfo } from '../types'
import { ConnectionsMap } from './ConnectionsMap'

interface Props {
  initialRef: string
  onChavruta: (ref: string) => void
}

/** The section a segment ref belongs to: "Berakhot 2a:3" -> "Berakhot 2a". */
export function sectionOf(ref: string): string {
  const i = ref.indexOf(':')
  return i === -1 ? ref : ref.slice(0, i)
}

function NoteEditor({ refName, onSaved }: { refName: string; onSaved: () => void }) {
  const { t } = useI18n()
  const [body, setBody] = useState('')
  const [open, setOpen] = useState(false)
  if (!open)
    return (
      <button type="button" className="link small" onClick={() => setOpen(true)}>
        + {t('addNote')}
      </button>
    )
  return (
    <form
      className="note-editor"
      onSubmit={(e) => {
        e.preventDefault()
        void post<Note>('/api/notes', { ref: refName, body }).then(() => {
          setBody('')
          setOpen(false)
          onSaved()
        })
      }}
    >
      <textarea
        value={body}
        onChange={(e) => setBody(e.target.value)}
        dir="auto"
        rows={2}
        aria-label={t('addNote')}
      />
      <button type="submit" disabled={!body.trim()}>
        {t('save')}
      </button>
      <button type="button" className="secondary" onClick={() => setOpen(false)}>
        {t('cancel')}
      </button>
    </form>
  )
}

function CardMaker({ refName }: { refName: string }) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const [front, setFront] = useState('')
  const [back, setBack] = useState('')
  const [suggested, setSuggested] = useState<{ front: string; back: string }[]>([])
  const [message, setMessage] = useState<string | null>(null)
  if (!open)
    return (
      <button type="button" className="link small" onClick={() => setOpen(true)}>
        + {t('makeCards')}
      </button>
    )
  const add = (f: string, b: string, origin: string) =>
    post<Card>('/api/cards', { ref: refName, front: f, back: b, origin }).then(() =>
      setMessage('✓'),
    )
  return (
    <div className="card-maker">
      <input
        value={front}
        onChange={(e) => setFront(e.target.value)}
        placeholder={t('question')}
        dir="auto"
      />
      <input
        value={back}
        onChange={(e) => setBack(e.target.value)}
        placeholder={t('answer')}
        dir="auto"
      />
      <button
        type="button"
        disabled={!front.trim() || !back.trim()}
        onClick={() => void add(front, back, 'manual')}
      >
        {t('addCard')}
      </button>
      <button
        type="button"
        className="secondary"
        onClick={() =>
          void post<{ front: string; back: string }[]>('/api/cards/suggest', { unit: refName })
            .then(setSuggested)
            .catch((e: unknown) => setMessage(e instanceof ApiError ? e.message : String(e)))
        }
      >
        {t('suggestCards')}
      </button>
      {suggested.map((s) => (
        <div key={s.front} className="suggested" dir="auto">
          <strong>{s.front}</strong> — {s.back}{' '}
          <button
            type="button"
            className="link small"
            onClick={() => void add(s.front, s.back, 'suggested')}
          >
            +
          </button>
        </div>
      ))}
      {message && <span className="muted">{message}</span>}
    </div>
  )
}

export function SugyaView({ initialRef, onChavruta }: Props) {
  const { t, lang } = useI18n()
  const [works, setWorks] = useState<WorkInfo[]>([])
  const [ref, setRef] = useState(sectionOf(initialRef))
  const [sugya, setSugya] = useState<Sugya | null>(null)
  const [links, setLinks] = useState<Links | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [show, setShow] = useState({ translation: true, Rashi: true, Tosafot: false })

  useEffect(() => {
    void api<WorkInfo[]>('/api/works')
      .then(setWorks)
      .catch(() => undefined)
  }, [])

  const load = (r: string) =>
    api<Sugya>(`/api/sugya?${query({ ref: r })}`)
      .then((s) => {
        setSugya(s)
        setError(null)
      })
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : String(e)))

  useEffect(() => {
    setLinks(null)
    void load(ref)
    void api<Links>(`/api/links?${query({ ref })}`)
      .then(setLinks)
      .catch(() => setLinks(null))
  }, [ref])

  const work = works.find((w) => w.title === sugya?.work)

  return (
    <div className="sugya">
      <div className="sugya-nav">
        <select
          value={sugya?.work ?? ''}
          onChange={(e) => {
            const w = works.find((x) => x.title === e.target.value)
            if (w?.sections[0]) setRef(w.sections[0])
          }}
          aria-label={t('works')}
        >
          {works.map((w) => (
            <option key={w.title} value={w.title}>
              {lang === 'he' ? w.he_title : w.title}
            </option>
          ))}
        </select>
        <select value={ref} onChange={(e) => setRef(e.target.value)} aria-label={t('learn')}>
          {(work?.sections ?? [ref]).map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="secondary"
          disabled={!sugya?.previous}
          onClick={() => sugya?.previous && setRef(sugya.previous)}
        >
          {t('previous')}
        </button>
        <button
          type="button"
          className="secondary"
          disabled={!sugya?.next}
          onClick={() => sugya?.next && setRef(sugya.next)}
        >
          {t('next')}
        </button>
        <label>
          <input
            type="checkbox"
            checked={show.translation}
            onChange={(e) => setShow({ ...show, translation: e.target.checked })}
          />
          {t('translation')}
        </label>
        <label>
          <input
            type="checkbox"
            checked={show.Rashi}
            onChange={(e) => setShow({ ...show, Rashi: e.target.checked })}
          />
          רש״י
        </label>
        <label>
          <input
            type="checkbox"
            checked={show.Tosafot}
            onChange={(e) => setShow({ ...show, Tosafot: e.target.checked })}
          />
          תוספות
        </label>
      </div>
      {error && <div className="notice notice-error">{error}</div>}
      {sugya && (
        <div className="sugya-body">
          <div className="segments">
            <h2>
              <bdi>{sugya.ref}</bdi>
            </h2>
            {sugya.segments.map((seg) => {
              const he = seg.texts.find((x) => x.language === 'he')
              const en = seg.texts.find((x) => x.language === 'en')
              const notes = sugya.notes.filter((n) => n.ref === seg.ref)
              const commentary = seg.commentary.filter(
                (c) =>
                  (c.work.startsWith('Rashi') && show.Rashi) ||
                  (c.work.startsWith('Tosafot') && show.Tosafot),
              )
              return (
                <article key={seg.ref} className="segment" id={seg.ref}>
                  <div className="segment-ref">
                    <bdi>{seg.ref.split(':').slice(-1)[0]}</bdi>
                  </div>
                  <div className="segment-texts">
                    {he && (
                      <p dir="rtl" className="hebrew">
                        {he.text}
                      </p>
                    )}
                    {en && show.translation && <p dir="ltr">{en.text}</p>}
                    {commentary.map((c) => (
                      <p key={c.ref + c.language} className="commentary" dir={dirOf(c.language)}>
                        <span className="commentary-tag">
                          {c.work.startsWith('Rashi') ? 'רש״י' : 'תוס׳'}
                        </span>
                        {c.text}
                      </p>
                    ))}
                    {notes.map((n) => (
                      <div key={n.id} className="note" dir="auto">
                        {n.body}
                        <button
                          type="button"
                          className="link small"
                          onClick={() => void del(`/api/notes/${n.id}`).then(() => load(ref))}
                        >
                          {t('delete')}
                        </button>
                        <button
                          type="button"
                          className="link small"
                          onClick={() => {
                            const body = window.prompt(t('edit'), n.body)
                            if (body)
                              void put(`/api/notes/${n.id}`, { ref: n.ref, body }).then(() =>
                                load(ref),
                              )
                          }}
                        >
                          {t('edit')}
                        </button>
                      </div>
                    ))}
                    <div className="segment-actions">
                      <NoteEditor refName={seg.ref} onSaved={() => void load(ref)} />
                      <CardMaker refName={seg.ref} />
                      <button
                        type="button"
                        className="link small"
                        onClick={() => onChavruta(seg.ref)}
                      >
                        📖 {t('learnThis')}
                      </button>
                    </div>
                  </div>
                </article>
              )
            })}
            <p className="attribution">
              {sugya.attribution.map((a) => `${a.version} (${a.license})`).join(' · ')}
            </p>
          </div>
          <aside className="side">
            <h3>{t('connections')}</h3>
            {links ? (
              <ConnectionsMap links={links} onOpen={(r) => setRef(sectionOf(r))} />
            ) : (
              <p className="muted">…</p>
            )}
          </aside>
        </div>
      )}
    </div>
  )
}
