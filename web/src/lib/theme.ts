import { useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

function systemTheme(): Theme {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function currentTheme(): Theme {
  const t = document.documentElement.dataset.theme as Theme | undefined
  return t ?? systemTheme()
}

/** Theme toggle; charts subscribe via useThemeVersion so they redraw with the new tokens. */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(currentTheme())
  const toggle = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark'
    document.documentElement.dataset.theme = next
    try { localStorage.setItem('theme', next) } catch { /* private mode */ }
    setTheme(next)
    window.dispatchEvent(new Event('themechange'))
  }
  return { theme, toggle }
}

export function useThemeVersion() {
  const [v, setV] = useState(0)
  useEffect(() => {
    const bump = () => setV((x) => x + 1)
    window.addEventListener('themechange', bump)
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)')
    mq?.addEventListener?.('change', bump)
    return () => {
      window.removeEventListener('themechange', bump)
      mq?.removeEventListener?.('change', bump)
    }
  }, [])
  return v
}

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}
