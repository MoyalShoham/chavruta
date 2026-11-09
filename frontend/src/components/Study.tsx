import { useEffect, useState } from 'react'
import { api, del, post } from '../api'
import { useI18n } from '../i18n'
import type { Card, DafYomi, PlanSummary, PlanToday, WorkInfo } from '../types'

function Review() {
  const { t } = useI18n()
  const [due, setDue] = useState<Card[]>([])
  const [revealed, setRevealed] = useState(false)
  const load = () => api<Card[]>('/api/cards?due=true').then(setDue)
  useEffect(() => {
    void load()
  }, [])
  const card = due[0]
  const grade = async (g: number) => {
    if (!card) return
    await post(`/api/cards/${card.id}/review`, { grade: g })
    setRevealed(false)
    await load()
  }
  return (
    <section className="panel">
      <h3>
        {t('dueCards')} ({due.length})
      </h3>
      {!card && <p className="muted">{t('noDue')}</p>}
      {card && (
        <div className="flashcard">
          <p className="muted">
            <bdi>{card.ref}</bdi>
          </p>
          <p className="front" dir="auto">
            {card.front}
          </p>
          {revealed ? (
            <>
              <p className="back" dir="auto">
                {card.back}
              </p>
              <div className="grades">
                <button type="button" className="again" onClick={() => void grade(1)}>
                  {t('again')}
                </button>
                <button type="button" className="secondary" onClick={() => void grade(3)}>
                  {t('hard')}
                </button>
                <button type="button" onClick={() => void grade(4)}>
                  {t('good')}
                </button>
                <button type="button" className="secondary" onClick={() => void grade(5)}>
                  {t('easy')}
                </button>
              </div>
            </>
          ) : (
            <button type="button" onClick={() => setRevealed(true)}>
              {t('show')}
            </button>
          )}
        </div>
      )}
    </section>
  )
}

function PlanCard({
  plan,
  onOpen,
  onDeleted,
}: {
  plan: PlanSummary
  onOpen: (ref: string) => void
  onDeleted: () => void
}) {
  const { t } = useI18n()
  const [today, setToday] = useState<PlanToday | null>(null)
  const load = () => api<PlanToday>(`/api/plans/${plan.id}/today`).then(setToday)
  useEffect(() => {
    void load()
  }, [plan.id])
  return (
    <div className="plan">
      <div className="plan-head">
        <strong>{plan.title}</strong>
        {today?.total ? (
          <span className="muted">
            {t('progress')}: {today.done}/{today.total}
          </span>
        ) : null}
        <button
          type="button"
          className="link small"
          onClick={() => void del(`/api/plans/${plan.id}`).then(onDeleted)}
        >
          {t('delete')}
        </button>
      </div>
      <ul>
        {today?.units.map((u) => (
          <li key={u.ref} className={u.done ? 'done' : ''}>
            <button type="button" className="link" onClick={() => onOpen(u.ref)}>
              <bdi>{u.ref}</bdi>
            </button>
            {!u.done && (
              <button
                type="button"
                className="secondary small"
                onClick={() =>
                  void post(`/api/plans/${plan.id}/progress`, { unit: u.ref }).then(load)
                }
              >
                ✓ {t('done')}
              </button>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}

export function Study({ onOpen }: { onOpen: (ref: string) => void }) {
  const { t, lang } = useI18n()
  const [daf, setDaf] = useState<DafYomi | null>(null)
  const [plans, setPlans] = useState<PlanSummary[]>([])
  const [works, setWorks] = useState<WorkInfo[]>([])
  const [work, setWork] = useState('')
  const [perDay, setPerDay] = useState(1)
  const loadPlans = () => api<PlanSummary[]>('/api/plans').then(setPlans)
  useEffect(() => {
    void api<DafYomi>('/api/daf-yomi')
      .then(setDaf)
      .catch(() => undefined)
    void loadPlans()
    void api<WorkInfo[]>('/api/works').then((w) => {
      setWorks(w)
      setWork(w[0]?.title ?? '')
    })
  }, [])
  return (
    <div className="study">
      <section className="panel">
        <h3>{t('dafYomi')}</h3>
        {daf?.ref ? (
          <p>
            <bdi className="big">{daf.ref}</bdi>{' '}
            {daf.in_library ? (
              <button type="button" className="link" onClick={() => daf.ref && onOpen(daf.ref)}>
                {t('learn')} →
              </button>
            ) : (
              <>
                <span className="muted">{t('notInLibrary')}</span>{' '}
                {daf.url && (
                  <a href={daf.url} target="_blank" rel="noreferrer noopener">
                    {t('openInSefaria')} ↗
                  </a>
                )}
              </>
            )}
          </p>
        ) : (
          <p className="muted">…</p>
        )}
        {!plans.some((p) => p.kind === 'daf_yomi') && (
          <button
            type="button"
            className="secondary"
            onClick={() => void post('/api/plans', { kind: 'daf_yomi' }).then(loadPlans)}
          >
            + {t('dafYomi')}
          </button>
        )}
      </section>
      <section className="panel">
        <h3>{t('plans')}</h3>
        {plans.map((p) => (
          <PlanCard key={p.id} plan={p} onOpen={onOpen} onDeleted={() => void loadPlans()} />
        ))}
        <form
          className="new-plan"
          onSubmit={(e) => {
            e.preventDefault()
            void post('/api/plans', { kind: 'work', work_title: work, per_day: perDay }).then(
              loadPlans,
            )
          }}
        >
          <strong>{t('newPlan')}:</strong>
          <select value={work} onChange={(e) => setWork(e.target.value)} aria-label={t('works')}>
            {works.map((w) => (
              <option key={w.title} value={w.title}>
                {lang === 'he' ? w.he_title : w.title}
              </option>
            ))}
          </select>
          <input
            type="number"
            min={1}
            max={50}
            value={perDay}
            onChange={(e) => setPerDay(Number(e.target.value))}
            aria-label={t('perDay')}
          />
          <span>{t('perDay')}</span>
          <button type="submit">+</button>
        </form>
      </section>
      <Review />
    </div>
  )
}
