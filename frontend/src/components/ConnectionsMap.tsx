import { useMemo } from 'react'
import { useI18n } from '../i18n'
import type { Links } from '../types'

const COLORS: Record<string, string> = {
  Commentary: '#b5651d',
  Tanakh: '#2e6f9e',
  Talmud: '#6b4c9a',
  Halakhah: '#2f855a',
  Mishnah: '#9b2c2c',
  Midrash: '#b7791f',
  Liturgy: '#4a5568',
}

const HE: Record<string, string> = {
  Commentary: 'מפרשים',
  Tanakh: 'תנ״ך',
  Talmud: 'תלמוד',
  Halakhah: 'הלכה',
  Mishnah: 'משנה',
  Midrash: 'מדרש',
  Liturgy: 'תפילה',
  Kabbalah: 'קבלה',
  Responsa: 'שו״ת',
  'Jewish Thought': 'מחשבה',
  Musar: 'מוסר',
  Chasidut: 'חסידות',
  Other: 'אחר',
}

/** A radial map: the passage in the middle, one spoke per category, sized by link count. */
export function ConnectionsMap({ links, onOpen }: { links: Links; onOpen: (ref: string) => void }) {
  const { lang, t } = useI18n()
  const groups = useMemo(
    () => Object.entries(links.groups).sort((a, b) => b[1].length - a[1].length),
    [links],
  )
  if (!groups.length) return <p className="muted">—</p>
  const size = 320
  const c = size / 2
  const max = Math.max(...groups.map(([, v]) => v.length))
  return (
    <div className="connections">
      <svg viewBox={`0 0 ${size} ${size}`} role="img" aria-label={t('connections')}>
        {groups.map(([cat, items], i) => {
          const angle = (2 * Math.PI * i) / groups.length - Math.PI / 2
          const r = 110
          const x = c + r * Math.cos(angle)
          const y = c + r * Math.sin(angle)
          const radius = 12 + 18 * Math.sqrt(items.length / max)
          const color = COLORS[cat] ?? '#718096'
          return (
            <g key={cat}>
              <line
                x1={c}
                y1={c}
                x2={x}
                y2={y}
                stroke={color}
                strokeWidth={1 + (items.length / max) * 4}
                opacity={0.5}
              />
              <circle cx={x} cy={y} r={radius} fill={color} opacity={0.85} />
              <text x={x} y={y + 4} textAnchor="middle" fontSize="11" fill="#fff" fontWeight="600">
                {items.length}
              </text>
              <text x={x} y={y + radius + 13} textAnchor="middle" fontSize="11" fill="currentColor">
                {lang === 'he' ? (HE[cat] ?? cat) : cat}
              </text>
            </g>
          )
        })}
        <circle cx={c} cy={c} r={30} fill="#1f3a5f" />
        <text x={c} y={c + 4} textAnchor="middle" fontSize="10" fill="#f4e9d0">
          {links.ref.split(' ').slice(-1)[0]}
        </text>
      </svg>
      <div className="connection-lists">
        {groups.map(([cat, items]) => (
          <details key={cat}>
            <summary>
              <span className="dot" style={{ background: COLORS[cat] ?? '#718096' }} />
              {lang === 'he' ? (HE[cat] ?? cat) : cat} ({items.length})
            </summary>
            <ul>
              {items.slice(0, 60).map((item) => (
                <li key={item.ref}>
                  {item.in_library ? (
                    <button type="button" className="link" onClick={() => onOpen(item.ref)}>
                      <bdi>{item.ref}</bdi> · {t('inLibrary')}
                    </button>
                  ) : (
                    <a href={item.url} target="_blank" rel="noreferrer noopener">
                      <bdi>{item.ref}</bdi> ↗
                    </a>
                  )}
                </li>
              ))}
            </ul>
          </details>
        ))}
      </div>
    </div>
  )
}
