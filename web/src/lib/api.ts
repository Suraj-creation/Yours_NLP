/* Thin fetch client for the FastAPI backend. Types stay loose on purpose: the JSON
   is produced by nlp_core/experiments.py and documented there. */
import { useMutation, useQuery } from '@tanstack/react-query'

export type Any = any // eslint-disable-line @typescript-eslint/no-explicit-any

async function request(path: string, init?: RequestInit): Promise<Any> {
  const res = await fetch(path, init)
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch { /* not json */ }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json()
}

export const api = {
  get: (path: string) => request(path),
  post: (path: string, body: unknown) =>
    request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  upload: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return request('/api/upload', { method: 'POST', body: fd })
  },
}

export function useResult(name: string) {
  return useQuery({ queryKey: ['result', name], queryFn: () => api.get(`/api/results/${name}`), staleTime: Infinity })
}

export function useMeta() {
  return useQuery({ queryKey: ['meta'], queryFn: () => api.get('/api/meta'), staleTime: Infinity })
}

export function useGet(path: string | null, key?: unknown[]) {
  return useQuery({ queryKey: key ?? ['get', path], queryFn: () => api.get(path as string), enabled: !!path, staleTime: 60_000 })
}

export function usePost<T = Any>(path: string) {
  return useMutation<Any, Error, T>({ mutationFn: (body: T) => api.post(path, body) })
}

export const fileUrl = (name: string) => `/api/files/${name}`
