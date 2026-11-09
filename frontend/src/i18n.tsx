import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import type { Lang } from './types'

const strings = {
  appName: { he: 'חברותא', en: 'Chavruta' },
  tagline: {
    he: 'שותפה ללימוד: תשובות מתוך המקורות, עם מראי מקום',
    en: 'A study partner: answers from the sources, with citations',
  },
  ask: { he: 'שאלה', en: 'Ask' },
  learn: { he: 'סוגיה', en: 'Text' },
  study: { he: 'לימוד יומי', en: 'Study plan' },
  notes: { he: 'הערות', en: 'Notes' },
  corpus: { he: 'ספרייה', en: 'Library' },
  signIn: { he: 'כניסה', en: 'Sign in' },
  signOut: { he: 'יציאה', en: 'Sign out' },
  username: { he: 'שם משתמש', en: 'Username' },
  password: { he: 'סיסמה', en: 'Password' },
  questionPlaceholder: {
    he: 'שאלו על התורה, הגמרא, רש״י ותוספות, הרמב״ם או השולחן ערוך; ציון ספר ומקום עוזר…',
    en: 'Ask about the Torah, the Gemara, Rashi and Tosafot, the Rambam or the Shulchan Arukh; naming the book and place helps…',
  },
  send: { he: 'שליחה', en: 'Send' },
  thinking: { he: 'מעיינת במקורות…', en: 'Looking in the sources…' },
  newConversation: { he: 'שיחה חדשה', en: 'New conversation' },
  history: { he: 'שיחות קודמות', en: 'Earlier conversations' },
  sources: { he: 'מקורות', en: 'Sources' },
  openInSefaria: { he: 'פתיחה בספריא', en: 'Open in Sefaria' },
  fetched: { he: 'מספריא', en: 'From Sefaria' },
  fetchedTitle: {
    he: 'נשלף מספריא לשאלה הזו; לא חלק מהספרייה',
    en: 'Looked up on Sefaria for this question; not part of the library',
  },
  license: { he: 'רישיון', en: 'License' },
  version: { he: 'מהדורה', en: 'Version' },
  samples: { he: 'שאלות לדוגמה', en: 'Sample questions' },
  chavrutaMode: { he: 'לימוד בחברותא', en: 'Chavruta mode' },
  chavrutaOn: {
    he: 'לומדים יחד על: ',
    en: 'Learning together on: ',
  },
  learnThis: { he: 'ללמוד בחברותא', en: 'Learn in chavruta' },
  commentaries: { he: 'מפרשים', en: 'Commentaries' },
  translation: { he: 'תרגום', en: 'Translation' },
  connections: { he: 'קשרים', en: 'Connections' },
  inLibrary: { he: 'בספרייה', en: 'In the library' },
  previous: { he: 'הקודם', en: 'Previous' },
  next: { he: 'הבא', en: 'Next' },
  addNote: { he: 'הוספת הערה', en: 'Add a note' },
  save: { he: 'שמירה', en: 'Save' },
  cancel: { he: 'ביטול', en: 'Cancel' },
  delete: { he: 'מחיקה', en: 'Delete' },
  edit: { he: 'עריכה', en: 'Edit' },
  makeCards: { he: 'כרטיסי חזרה', en: 'Review cards' },
  suggestCards: { he: 'הצעת כרטיסים', en: 'Suggest cards' },
  addCard: { he: 'הוספת כרטיס', en: 'Add card' },
  question: { he: 'שאלה', en: 'Question' },
  answer: { he: 'תשובה', en: 'Answer' },
  show: { he: 'הצגת התשובה', en: 'Show answer' },
  again: { he: 'שוב', en: 'Again' },
  hard: { he: 'קשה', en: 'Hard' },
  good: { he: 'טוב', en: 'Good' },
  easy: { he: 'קל', en: 'Easy' },
  dueCards: { he: 'כרטיסים לחזרה היום', en: 'Cards due today' },
  noDue: { he: 'אין כרטיסים לחזרה היום.', en: 'No cards due today.' },
  dafYomi: { he: 'הדף היומי', en: 'Daf Yomi' },
  notInLibrary: {
    he: 'הדף הזה אינו בספרייה; אפשר לפתוח אותו בספריא.',
    en: 'This daf is not in the library; open it in Sefaria.',
  },
  plans: { he: 'תוכניות לימוד', en: 'Study plans' },
  newPlan: { he: 'תוכנית חדשה', en: 'New plan' },
  perDay: { he: 'ביום', en: 'per day' },
  today: { he: 'היום', en: 'Today' },
  done: { he: 'סיימתי', en: 'Done' },
  progress: { he: 'התקדמות', en: 'Progress' },
  closedTitle: { he: 'שבת שלום', en: 'Shabbat Shalom' },
  closedYomTov: { he: 'חג שמח', en: 'Chag Sameach' },
  closedText: {
    he: 'חברותא סגורה בשבת ובחג. נתראה במוצאי',
    en: 'Chavruta is closed on Shabbat and Yom Tov. See you after',
  },
  opensAt: { he: 'נפתחת שוב ב־', en: 'Opens again at ' },
  timesFallback: {
    he: 'זמני השבת אינם זמינים כרגע, ולכן חברותא סגורה מיום שישי בצהריים.',
    en: 'Shabbat times are unavailable right now, so Chavruta closes from Friday noon.',
  },
  starting: {
    he: 'הספרייה נטענת. בהפעלה הראשונה זה לוקח זמן; הדף יתעדכן לבד.',
    en: 'The library is loading. The first start takes a while; this page updates by itself.',
  },
  works: { he: 'חיבורים', en: 'Works' },
  enabled: { he: 'פעיל', en: 'Enabled' },
  versions: { he: 'מהדורות', en: 'Versions' },
  included: { he: 'נכלל', en: 'Included' },
  skipped: { he: 'לא נכלל', en: 'Not included' },
  chunks: { he: 'יחידות', en: 'Units' },
  embedded: { he: 'מוטמעות', en: 'Embedded' },
  strictness: { he: 'מסנן נושאים', en: 'Scope filter' },
  strict: { he: 'מחמיר', en: 'Strict' },
  balanced: { he: 'מאוזן', en: 'Balanced' },
  lenient: { he: 'מקל', en: 'Lenient' },
  llmMode: { he: 'מצב המודל', en: 'Model mode' },
  licenseNote: {
    he: 'רק מהדורות ברישיון פתוח (נחלת הכלל, CC0, CC-BY, CC-BY-SA) נשמרות ומוצגות.',
    en: 'Only openly licensed versions (Public Domain, CC0, CC-BY, CC-BY-SA) are stored and shown.',
  },
  noNotes: { he: 'עדיין אין הערות.', en: 'No notes yet.' },
  unverified: { he: 'לא אומת', en: 'unverified' },
  uncited: { he: 'קטע זה אינו מצוטט ממקור', en: 'This part is not tied to a cited source' },
  uncitedLegend: {
    he: 'טקסט עם קו מקווקו אינו מצוטט ממקור',
    en: 'Text with a dotted underline is not tied to a cited source',
  },
  error: { he: 'שגיאה', en: 'Error' },
  language: { he: 'English', en: 'עברית' },
} satisfies Record<string, Record<Lang, string>>

export type Key = keyof typeof strings

interface I18n {
  lang: Lang
  setLang: (l: Lang) => void
  t: (k: Key) => string
}

const Ctx = createContext<I18n>({ lang: 'he', setLang: () => undefined, t: (k) => strings[k].he })

function initial(): Lang {
  try {
    const saved = localStorage.getItem('chavruta-lang')
    if (saved === 'he' || saved === 'en') return saved
  } catch {
    /* storage unavailable */
  }
  return 'he'
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(initial)
  useEffect(() => {
    document.documentElement.lang = lang
    document.documentElement.dir = lang === 'he' ? 'rtl' : 'ltr'
    try {
      localStorage.setItem('chavruta-lang', lang)
    } catch {
      /* storage unavailable */
    }
  }, [lang])
  return (
    <Ctx.Provider value={{ lang, setLang, t: (k) => strings[k][lang] }}>{children}</Ctx.Provider>
  )
}

export const useI18n = () => useContext(Ctx)

/** Hebrew text gets dir="rtl", Latin dir="ltr", whatever the page direction. */
export function dirOf(language: string): 'rtl' | 'ltr' {
  return language === 'he' ? 'rtl' : 'ltr'
}
