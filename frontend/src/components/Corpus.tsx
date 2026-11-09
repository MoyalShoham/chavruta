import { useEffect, useState } from 'react'
import { api, ApiError, post } from '../api'
import { useI18n } from '../i18n'
import type { CorpusOverview } from '../types'

export function Corpus() {
  const { t, lang } = useI18n()
  const [data, setData] = useState<CorpusOverview | null>(null)
  const [error, setError] = useState<string | null>(null)
  const load = () =>
    api<CorpusOverview>('/api/corpus')
      .then(setData)
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : String(e)))
  useEffect(() => {
    void load()
  }, [])
  if (error) return <div className="notice notice-error">{error}</div>
  if (!data) return <p className="muted">…</p>
  return (
    <div className="corpus">
      <div className="corpus-head">
        <span>
          {data.library.chunks.toLocaleString()} {t('chunks')} · {data.works.length} {t('works')}
        </span>
        <label>
          {t('strictness')}:{' '}
          <select
            value={data.strictness}
            onChange={(e) =>
              void post('/api/corpus/strictness', { strictness: e.target.value }).then(load)
            }
          >
            <option value="strict">{t('strict')}</option>
            <option value="balanced">{t('balanced')}</option>
            <option value="lenient">{t('lenient')}</option>
          </select>
        </label>
        <span>
          {t('llmMode')}: <code>{data.llm_mode}</code>
        </span>
      </div>
      <p className="muted">{t('licenseNote')}</p>
      <table>
        <thead>
          <tr>
            <th>{t('works')}</th>
            <th>{t('chunks')}</th>
            <th>{t('embedded')}</th>
            <th>{t('versions')}</th>
            <th>{t('enabled')}</th>
          </tr>
        </thead>
        <tbody>
          {data.works.map((w) => (
            <tr key={w.id} className={w.enabled ? '' : 'disabled'}>
              <td>
                <strong>{lang === 'he' ? w.he_title : w.title}</strong>
                <div className="muted small">{w.category}</div>
              </td>
              <td>
                {Object.entries(w.chunks)
                  .map(([l, n]) => `${l}: ${n}`)
                  .join(' · ')}
              </td>
              <td>{w.embedded}</td>
              <td>
                <ul className="versions">
                  {w.versions.map((v) => (
                    <li key={v.language + v.title} className={v.included ? 'included' : 'skipped'}>
                      <span className="badge">{v.included ? t('included') : t('skipped')}</span>{' '}
                      <bdi>{v.title}</bdi>{' '}
                      <span className="muted">
                        ({v.language}, {v.license})
                      </span>
                      {v.reason && <div className="muted small">{v.reason}</div>}
                    </li>
                  ))}
                </ul>
              </td>
              <td>
                <input
                  type="checkbox"
                  checked={w.enabled}
                  aria-label={`${t('enabled')} ${w.title}`}
                  onChange={(e) =>
                    void post(`/api/corpus/works/${w.id}/enabled`, {
                      enabled: e.target.checked,
                    }).then(load)
                  }
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
