import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../App'
import { AnswerView, emphasis } from '../components/AnswerView'
import { Corpus } from '../components/Corpus'
import { sectionOf } from '../components/SugyaView'
import { I18nProvider } from '../i18n'
import type { Answer, CorpusOverview } from '../types'

const answer: Answer = {
  status: 'answered',
  segments: [
    {
      text: 'One whose dead lies before him is exempt from the Shema.',
      citations: [
        { source_index: 0, ref: 'Mishnah Berakhot 3:1', language: 'he', cited_text: 'x' },
      ],
    },
    { text: ' The Mishnah Berurah adds more.', citations: [] },
  ],
  notices: [
    { kind: 'not_psak', en: 'Not a posek.', he: 'אינה פוסקת.' },
    { kind: 'ask_rabbi_first', en: 'Please ask your rabbi first.', he: 'יש לשאול רב.' },
    { kind: 'unretrieved_mention', en: 'Names an unretrieved source.', he: 'מקור שלא נשלף.' },
  ],
  sources: [
    {
      index: 0,
      ref: 'Mishnah Berakhot 3:1',
      work: 'Mishnah Berakhot',
      category: 'Mishnah',
      url: 'https://www.sefaria.org/Mishnah_Berakhot.3.1',
      texts: [
        {
          language: 'he',
          version: 'Torat Emet 357',
          license: 'Public Domain',
          text: 'מי שמתו מוטל לפניו',
        },
        { language: 'en', version: 'Mishnah Yomit', license: 'CC-BY', text: 'One whose dead...' },
      ],
    },
    {
      index: 1,
      ref: 'Berakhot 17b:1',
      work: 'Berakhot',
      category: 'Talmud',
      url: 'https://www.sefaria.org/Berakhot.17b.1',
      texts: [],
    },
  ],
  unretrieved: ['Mishnah Berurah'],
  decision: {
    reason: 'ok',
    sensitive_area: 'mourning',
    scope: 'in_scope',
    kind: 'practical_halacha',
  },
  model: 'claude-opus-5-5',
}

function json(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }),
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('AnswerView', () => {
  it('puts "ask your rabbi" above the answer and disclaimers below it', () => {
    render(
      <I18nProvider>
        <AnswerView answer={answer} />
      </I18nProvider>,
    )
    const rabbi = screen.getByText('יש לשאול רב.')
    const text = screen.getByText(/One whose dead lies/)
    const psak = screen.getByText('אינה פוסקת.')
    expect(rabbi.compareDocumentPosition(text) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(text.compareDocumentPosition(psak) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(screen.getByText('מקור שלא נשלף.')).toBeInTheDocument()
  })

  it('opens the cited source with its version and license', async () => {
    render(
      <I18nProvider>
        <AnswerView answer={answer} />
      </I18nProvider>,
    )
    await userEvent.click(screen.getByRole('button', { name: '1' }))
    const source = document.getElementById('source-0')
    expect(source).not.toBeNull()
    const s = within(source as HTMLElement)
    expect(s.getByText('מי שמתו מוטל לפניו')).toHaveAttribute('dir', 'rtl')
    expect(s.getByText(/Public Domain/)).toBeInTheDocument()
    expect(s.getByText(/CC-BY/)).toBeInTheDocument()
    expect(s.getByRole('link')).toHaveAttribute(
      'href',
      'https://www.sefaria.org/Mishnah_Berakhot.3.1',
    )
  })

  it('marks text that is not tied to a source', () => {
    render(
      <I18nProvider>
        <AnswerView answer={answer} />
      </I18nProvider>,
    )
    expect(screen.getByText(/The Mishnah Berurah adds more/)).toHaveClass('uncited')
    expect(screen.getByText(/One whose dead lies/)).toHaveClass('cited')
    expect(screen.getByText('טקסט עם קו מקווקו אינו מצוטט ממקור')).toBeInTheDocument()
  })

  it('shows **bold** as bold, also across citation segments, and never invents emphasis', () => {
    expect(emphasis(['a *', '*b** c'])).toEqual([
      [{ text: 'a ', bold: false }],
      [
        { text: 'b', bold: true },
        { text: ' c', bold: false },
      ],
    ])
    expect(emphasis(['**The sources', ' say** this', ' and 2**3'])).toEqual([
      [{ text: 'The sources', bold: true }],
      [
        { text: ' say', bold: true },
        { text: ' this', bold: false },
      ],
      [{ text: ' and 2**3', bold: false }],
    ])
    const bolded: Answer = {
      ...answer,
      segments: [{ text: '**Not in the sources.** Here is why.', citations: [] }],
    }
    render(
      <I18nProvider>
        <AnswerView answer={bolded} />
      </I18nProvider>,
    )
    expect(screen.getByText('Not in the sources.').tagName).toBe('STRONG')
    expect(screen.queryByText(/\*\*/)).toBeNull()
  })

  it('marks a source looked up on Sefaria for this question', () => {
    const fetched: Answer = {
      ...answer,
      sources: answer.sources.map((src, i) =>
        i === 0 ? { ...src, ref: 'Bava Batra 5a:1', fetched: true } : src,
      ),
      notices: [{ kind: 'fetched', en: 'Looked up on Sefaria.', he: 'נשלף מספריא.' }],
    }
    render(
      <I18nProvider>
        <AnswerView answer={fetched} />
      </I18nProvider>,
    )
    const badge = screen.getByText('מספריא')
    expect(badge).toHaveAttribute('title', 'נשלף מספריא לשאלה הזו; לא חלק מהספרייה')
    expect(screen.getAllByText('מספריא')).toHaveLength(1)
    expect(screen.getByText('נשלף מספריא.')).toHaveClass('notice-fetched')
  })

  it('lists cited sources first and counts them', () => {
    render(
      <I18nProvider>
        <AnswerView answer={answer} />
      </I18nProvider>,
    )
    expect(screen.getByText(/מקורות \(1\/2\)/)).toBeInTheDocument()
  })
})

describe('App', () => {
  it('shows the closed screen on Shabbat and makes no other request', async () => {
    const fetch = vi.fn((url: string) =>
      url === '/api/status'
        ? json({
            closed: true,
            until: '2027-04-24T19:52:00+03:00',
            reason: 'Shabbat',
            reason_he: 'שבת',
            source: 'hebcal',
          })
        : json({}, 500),
    )
    vi.stubGlobal('fetch', fetch)
    render(
      <I18nProvider>
        <App />
      </I18nProvider>,
    )
    expect(await screen.findByText('שבת שלום')).toBeInTheDocument()
    expect(fetch.mock.calls.map((c) => c[0])).toEqual(['/api/status'])
  })

  it('shows a loading screen while the API is still starting', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => json({ detail: 'The API is not responding yet' }, 503)),
    )
    render(
      <I18nProvider>
        <App />
      </I18nProvider>,
    )
    expect(await screen.findByText(/הספרייה נטענת/)).toBeInTheDocument()
  })

  it('closes at once when any request answers 423', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (url === '/api/status')
          return json({
            closed: false,
            until: null,
            reason: null,
            reason_he: null,
            source: 'hebcal',
          })
        return json(
          {
            detail: 'closed',
            closed: true,
            until: '2027-04-22T19:51:00+03:00',
            reason: 'Pesach I',
            reason_he: 'פסח א׳',
            source: 'hebcal',
          },
          423,
        )
      }),
    )
    render(
      <I18nProvider>
        <App />
      </I18nProvider>,
    )
    expect(await screen.findByText('חג שמח')).toBeInTheDocument()
  })

  it('is Hebrew and right-to-left first, and switches to English', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) =>
        url === '/api/status'
          ? json({ closed: false, until: null, reason: null, reason_he: null, source: 'hebcal' })
          : json({ detail: 'sign in first' }, 401),
      ),
    )
    render(
      <I18nProvider>
        <App />
      </I18nProvider>,
    )
    await screen.findByRole('heading', { name: 'כניסה' })
    expect(document.documentElement.dir).toBe('rtl')
    await userEvent.click(screen.getByRole('button', { name: 'English' }))
    await waitFor(() => expect(document.documentElement.dir).toBe('ltr'))
    expect(screen.getByRole('heading', { name: 'Sign in' })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'עברית' }))
  })
})

describe('Corpus dashboard', () => {
  it('shows skipped versions with the reason', async () => {
    const data: CorpusOverview = {
      works: [
        {
          id: 1,
          title: 'Berakhot',
          he_title: 'ברכות',
          category: 'Talmud',
          enabled: true,
          synced_at: null,
          chunks: { he: 10, en: 10 },
          embedded: '20/20',
          versions: [
            {
              language: 'he',
              title: 'Wikisource Talmud Bavli',
              license: 'CC-BY-SA',
              included: true,
              reason: null,
            },
            {
              language: 'en',
              title: 'William Davidson Edition',
              license: 'CC-BY-NC',
              included: false,
              reason: 'license CC-BY-NC is not open',
            },
          ],
        },
      ],
      library: { chunks: 20, refs: 10 },
      strictness: 'balanced',
      relevance_threshold: 0.05,
      llm_mode: 'replay',
    }
    vi.stubGlobal(
      'fetch',
      vi.fn(() => json(data)),
    )
    render(
      <I18nProvider>
        <Corpus />
      </I18nProvider>,
    )
    expect(await screen.findByText('license CC-BY-NC is not open')).toBeInTheDocument()
    expect(screen.getByText('לא נכלל')).toBeInTheDocument()
  })
})

describe('sectionOf', () => {
  it('maps a segment to its section', () => {
    expect(sectionOf('Berakhot 2a:3')).toBe('Berakhot 2a')
    expect(sectionOf('Rashi on Berakhot 2a:1:1')).toBe('Rashi on Berakhot 2a')
    expect(sectionOf('Deuteronomy 6')).toBe('Deuteronomy 6')
  })
})
