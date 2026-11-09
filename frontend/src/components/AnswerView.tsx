import { useState } from 'react'
import { dirOf, useI18n } from '../i18n'
import type { Answer, Notice, Source } from '../types'

const BEFORE = new Set<Notice['kind']>(['safety', 'ask_rabbi_first'])

/** Connective text ("; ", ".") is not a claim; a segment with words and no citation is shown as such. */
const hasWords = (text: string) => /[\p{L}]{3,}/u.test(text)

type Run = { text: string; bold: boolean }

/** The model marks emphasis with **…**, and both a bold span and a single marker can cross the
 * segments that citations cut the answer into. Markers are found and paired in the whole answer;
 * an unpaired one is shown as text. */
export function emphasis(segments: string[]): Run[][] {
  const whole = segments.join('')
  const markers: number[] = []
  for (let i = whole.indexOf('**'); i !== -1; i = whole.indexOf('**', i + 2)) markers.push(i)
  const paired = markers.slice(0, markers.length - (markers.length % 2))
  const hidden = new Set(paired.flatMap((m) => [m, m + 1]))
  const bold = Array.from({ length: whole.length }, () => false)
  for (let k = 0; k < paired.length; k += 2) {
    for (let c = (paired[k] ?? 0) + 2; c < (paired[k + 1] ?? 0); c++) bold[c] = true
  }
  let offset = 0
  return segments.map((text) => {
    const runs: Run[] = []
    for (let c = offset; c < offset + text.length; c++) {
      if (hidden.has(c)) continue
      const b = bold[c] ?? false
      const last = runs[runs.length - 1]
      if (last && last.bold === b) last.text += whole[c]
      else runs.push({ text: whole[c] ?? '', bold: b })
    }
    offset += text.length
    return runs
  })
}

function NoticeBox({ notice }: { notice: Notice }) {
  const { lang } = useI18n()
  return (
    <div
      className={`notice notice-${notice.kind}`}
      role={notice.kind === 'safety' ? 'alert' : 'note'}
    >
      {lang === 'he' ? notice.he : notice.en}
    </div>
  )
}

function SourceCard({
  source,
  open,
  onToggle,
}: {
  source: Source
  open: boolean
  onToggle: () => void
}) {
  const { t } = useI18n()
  return (
    <li className={`source ${open ? 'open' : ''}`} id={`source-${source.index}`}>
      <button type="button" className="source-head" onClick={onToggle} aria-expanded={open}>
        <span className="source-num">{source.index + 1}</span>
        <bdi className="source-ref">{source.ref}</bdi>
        {source.fetched && (
          <span className="source-fetched" title={t('fetchedTitle')}>
            {t('fetched')}
          </span>
        )}
        <span className="source-work">{source.category}</span>
      </button>
      {open && (
        <div className="source-body">
          {source.texts.map((text) => (
            <div key={text.language + text.version} className="source-text">
              <p dir={dirOf(text.language)} className={text.language === 'he' ? 'hebrew' : ''}>
                {text.text}
              </p>
              <p className="attribution">
                {t('version')}: <bdi>{text.version}</bdi> · {t('license')}: {text.license}
              </p>
            </div>
          ))}
          <a href={source.url} target="_blank" rel="noreferrer noopener">
            {t('openInSefaria')} ↗
          </a>
        </div>
      )}
    </li>
  )
}

export function AnswerView({ answer }: { answer: Answer }) {
  const { t, lang } = useI18n()
  const [open, setOpen] = useState<Set<number>>(new Set())
  const toggle = (i: number) =>
    setOpen((prev) => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })
  const before = answer.notices.filter((n) => BEFORE.has(n.kind))
  const after = answer.notices.filter((n) => !BEFORE.has(n.kind))
  const cited = new Set(answer.segments.flatMap((s) => s.citations.map((c) => c.source_index)))
  const textDir = answer.segments.length ? undefined : lang === 'he' ? 'rtl' : 'ltr'
  const runs = emphasis(answer.segments.map((s) => s.text))

  return (
    <div className="answer">
      {before.map((n) => (
        <NoticeBox key={n.kind} notice={n} />
      ))}
      {answer.segments.length > 0 && (
        <div className="answer-text" dir="auto">
          {answer.segments.map((segment, i) => (
            <span
              key={i}
              className={
                segment.citations.length ? 'cited' : hasWords(segment.text) ? 'uncited' : ''
              }
              title={!segment.citations.length && hasWords(segment.text) ? t('uncited') : undefined}
            >
              {(runs[i] ?? []).map((run, j) =>
                run.bold ? <strong key={j}>{run.text}</strong> : run.text,
              )}
              {[...new Set(segment.citations.map((c) => c.source_index))].map((idx) => (
                <button
                  key={idx}
                  type="button"
                  className="cite"
                  title={answer.sources[idx]?.ref}
                  onClick={() => {
                    setOpen((prev) => new Set(prev).add(idx))
                    document.getElementById(`source-${idx}`)?.scrollIntoView?.({ block: 'nearest' })
                  }}
                >
                  {idx + 1}
                </button>
              ))}
            </span>
          ))}
        </div>
      )}
      {answer.segments.some((s) => !s.citations.length && hasWords(s.text)) && (
        <p className="legend">
          <span className="uncited">{t('uncitedLegend')}</span>
        </p>
      )}
      {after.map((n) => (
        <NoticeBox key={n.kind} notice={n} />
      ))}
      {answer.sources.length > 0 && (
        <details className="sources" open={cited.size > 0} dir={textDir}>
          <summary>
            {t('sources')} ({cited.size}/{answer.sources.length})
          </summary>
          <ol>
            {answer.sources
              .filter((s) => cited.has(s.index))
              .concat(answer.sources.filter((s) => !cited.has(s.index)))
              .map((s) => (
                <SourceCard
                  key={s.index}
                  source={s}
                  open={open.has(s.index)}
                  onToggle={() => toggle(s.index)}
                />
              ))}
          </ol>
        </details>
      )}
    </div>
  )
}
