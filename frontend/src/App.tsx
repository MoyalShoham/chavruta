import { useEffect, useState } from 'react'
import { api, ApiError, del, onClosed } from './api'
import { Ask } from './components/Ask'
import { Corpus } from './components/Corpus'
import { Closed, Login, Notes } from './components/Simple'
import { Study } from './components/Study'
import { SugyaView } from './components/SugyaView'
import { useI18n } from './i18n'
import type { ClosedStatus, User } from './types'

type Tab = 'ask' | 'learn' | 'study' | 'notes' | 'corpus'

export function App() {
  const { t, lang, setLang } = useI18n()
  const [status, setStatus] = useState<ClosedStatus | null>(null)
  const [user, setUser] = useState<User | null | undefined>(undefined)
  const [tab, setTab] = useState<Tab>('ask')
  const [sugyaRef, setSugyaRef] = useState('Berakhot 2a')
  const [anchor, setAnchor] = useState<string | null>(null)

  useEffect(() => {
    const check = () =>
      api<ClosedStatus>('/api/status')
        .then(setStatus)
        .catch(() => undefined)
    void check()
    const timer = window.setInterval(check, 60_000)
    const off = onClosed(setStatus)
    return () => {
      window.clearInterval(timer)
      off()
    }
  }, [])

  const isOpen = status !== null && !status.closed
  useEffect(() => {
    // Nothing is requested until the status says the app is open.
    if (!isOpen) return
    api<User>('/api/me')
      .then(setUser)
      .catch((e: unknown) => {
        if (e instanceof ApiError && e.status === 401) setUser(null)
      })
  }, [isOpen])

  if (status?.closed) return <Closed status={status} />

  const open = (ref: string) => {
    setSugyaRef(ref)
    setTab('learn')
  }
  const tabs: Tab[] =
    user?.role === 'ADMIN'
      ? ['ask', 'learn', 'study', 'notes', 'corpus']
      : ['ask', 'learn', 'study', 'notes']

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo">{t('appName')}</span>
          <span className="tagline">{t('tagline')}</span>
        </div>
        {user && (
          <nav aria-label="main">
            {tabs.map((k) => (
              <button
                key={k}
                type="button"
                className={tab === k ? 'tab active' : 'tab'}
                aria-current={tab === k ? 'page' : undefined}
                onClick={() => setTab(k)}
              >
                {t(k)}
              </button>
            ))}
          </nav>
        )}
        <div className="header-actions">
          <button
            type="button"
            className="link"
            onClick={() => setLang(lang === 'he' ? 'en' : 'he')}
          >
            {t('language')}
          </button>
          {user && (
            <button
              type="button"
              className="link"
              onClick={() => void del('/api/session').then(() => setUser(null))}
            >
              {t('signOut')}
            </button>
          )}
        </div>
      </header>
      <main>
        {user === null && <Login onSignedIn={setUser} />}
        {user && tab === 'ask' && (
          <Ask anchorRef={anchor} onLeaveChavruta={() => setAnchor(null)} />
        )}
        {user && tab === 'learn' && (
          <SugyaView
            initialRef={sugyaRef}
            onChavruta={(ref) => {
              setAnchor(ref)
              setTab('ask')
            }}
          />
        )}
        {user && tab === 'study' && <Study onOpen={open} />}
        {user && tab === 'notes' && <Notes onOpen={open} />}
        {user && tab === 'corpus' && <Corpus />}
      </main>
      <footer>
        {lang === 'he'
          ? 'חברותא היא שותפה ללימוד ואינה פוסקת הלכה. לשאלות למעשה יש לפנות לרב. הטקסטים מספריא, ברישיון פתוח.'
          : 'Chavruta is a study partner, not a posek. For practical questions, ask a rabbi. Texts from Sefaria, openly licensed.'}
      </footer>
    </div>
  )
}
