import type { ClosedStatus } from './types'

export class ApiError extends Error {
  status: number
  closed: ClosedStatus | null

  constructor(status: number, message: string, closed: ClosedStatus | null = null) {
    super(message)
    this.status = status
    this.closed = closed
  }
}

type ClosedListener = (status: ClosedStatus) => void
const closedListeners = new Set<ClosedListener>()

/** Called when any request comes back 423: the app closes for Shabbat or Yom Tov at once. */
export function onClosed(listener: ClosedListener): () => void {
  closedListeners.add(listener)
  return () => closedListeners.delete(listener)
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const method = init.method ?? 'GET'
  const headers = new Headers(init.headers)
  if (method !== 'GET') headers.set('X-Chavruta-Client', 'web')
  if (init.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  const response = await fetch(path, { ...init, headers, credentials: 'same-origin' })
  if (response.status === 204) return undefined as T
  let body: unknown = null
  try {
    body = await response.json()
  } catch {
    body = null
  }
  if (!response.ok) {
    const detail =
      body && typeof body === 'object' && 'detail' in body
        ? String((body as { detail: unknown }).detail)
        : `HTTP ${response.status}`
    if (response.status === 423) {
      const closed = body as ClosedStatus
      closedListeners.forEach((l) => l({ ...closed, closed: true }))
      throw new ApiError(423, detail, closed)
    }
    throw new ApiError(response.status, detail)
  }
  return body as T
}

export const post = <T>(path: string, data: unknown) =>
  api<T>(path, { method: 'POST', body: JSON.stringify(data) })
export const put = <T>(path: string, data: unknown) =>
  api<T>(path, { method: 'PUT', body: JSON.stringify(data) })
export const del = (path: string) => api<void>(path, { method: 'DELETE' })

export const query = (params: Record<string, string>) => new URLSearchParams(params).toString()
