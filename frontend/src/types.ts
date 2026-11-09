export type Lang = 'he' | 'en'

export interface User {
  id: number
  username: string
  display_name: string
  role: 'LEARNER' | 'ADMIN'
}

export interface ClosedStatus {
  closed: boolean
  until: string | null
  reason: string | null
  reason_he: string | null
  source: string
}

export interface Citation {
  source_index: number
  ref: string
  language: string
  cited_text: string
}

export interface Segment {
  text: string
  citations: Citation[]
}

export interface Notice {
  kind: 'safety' | 'ask_rabbi_first' | 'not_psak' | 'unretrieved_mention' | 'refusal' | 'no_source'
  en: string
  he: string
}

export interface SourceText {
  language: string
  version: string
  license: string
  text: string
}

export interface Source {
  index: number
  ref: string
  work: string
  category: string
  url: string
  texts: SourceText[]
}

export interface Answer {
  conversation_id?: number
  status: 'answered' | 'refused' | 'no_source'
  segments: Segment[]
  notices: Notice[]
  sources: Source[]
  unretrieved: string[]
  decision: { reason: string; sensitive_area: string | null; scope: string; kind: string } | null
  model: string | null
}

export interface ConversationMessage {
  role: 'user' | 'assistant'
  text: string
  answer: Answer | null
}

export interface Conversation {
  id: number
  mode: 'ask' | 'chavruta'
  anchor_ref: string | null
  title: string
  messages: ConversationMessage[]
}

export interface ConversationSummary {
  id: number
  mode: string
  anchor_ref: string | null
  title: string
  created_at: string
}

export interface SugyaText {
  language: string
  text: string
  version: string
  license: string
}

export interface Commentary {
  ref: string
  work: string
  language: string
  text: string
  license: string
}

export interface Note {
  id: number
  ref: string
  body: string
  updated_at: string
}

export interface Sugya {
  ref: string
  work: string
  segments: { ref: string; texts: SugyaText[]; commentary: Commentary[] }[]
  notes: Note[]
  previous: string | null
  next: string | null
  attribution: { language: string; version: string; license: string }[]
}

export interface WorkInfo {
  title: string
  he_title: string
  category: string
  sections: string[]
}

export interface LinkItem {
  ref: string
  anchor: string
  type: string
  in_library: boolean
  url: string
}

export interface Links {
  ref: string
  groups: Record<string, LinkItem[]>
  total: number
}

export interface Card {
  id: number
  ref: string
  front: string
  back: string
  due: string | null
  interval: number
  repetitions: number
  ease: number
  origin: string
}

export interface PlanSummary {
  id: number
  kind: 'daf_yomi' | 'work'
  title: string
  start_date: string
}

export interface PlanToday {
  id: number
  title: string
  kind: string
  units: { ref: string; done: boolean }[]
  done: number
  total: number | null
}

export interface DafYomi {
  date: string
  ref: string | null
  in_library: boolean
  url: string | null
}

export interface VersionInfo {
  language: string
  title: string
  license: string
  included: boolean
  reason: string | null
}

export interface CorpusWork {
  id: number
  title: string
  he_title: string
  category: string
  enabled: boolean
  synced_at: string | null
  chunks: Record<string, number>
  embedded: string
  versions: VersionInfo[]
}

export interface CorpusOverview {
  works: CorpusWork[]
  library: { chunks: number; refs: number }
  strictness: 'strict' | 'balanced' | 'lenient'
  relevance_threshold: number
  llm_mode: string
}
