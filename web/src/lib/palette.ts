/* Colour follows the entity, never its rank: every pipeline run and every corpus layer
   has one fixed categorical slot across the whole site. */
import { cssVar } from './theme'

export const RUN_SLOT: Record<string, number> = { A: 1, B: 2, H: 3, C1: 4, C2: 5, C3: 6, H2: 7 }
export const LAYER_SLOT: Record<string, number> = { concept: 1, research: 2, learner: 3, synthetic: 4 }
export const TOKENIZER_SLOT: Record<string, number> = {
  nltk: 1, spacy: 2, custom: 3, hybrid: 4, whitespace: 5, nltk_wordpunct: 6, nltk_treebank: 7, nltk_tweet: 8,
}

export const slot = (n: number) => cssVar(`--s${n}`)
export const runColor = (run: string) => slot(RUN_SLOT[run] ?? 8)
export const layerColor = (layer: string) => slot(LAYER_SLOT[layer] ?? 8)

export function chrome() {
  return {
    ink: cssVar('--ink'), ink2: cssVar('--ink-2'), ink3: cssVar('--ink-3'), line: cssVar('--line'),
    lineStrong: cssVar('--line-strong'), surface: cssVar('--surface'), surface2: cssVar('--surface-2'),
    accent: cssVar('--accent'), good: cssVar('--good'), bad: cssVar('--bad'), warn: cssVar('--warn'),
    seq: [0, 1, 2, 3, 4, 5, 6, 7].map((i) => cssVar(`--seq-${i}`)),
  }
}

export const fmt = (n: number | null | undefined, d = 0) =>
  n === null || n === undefined || Number.isNaN(n) ? '—' : Number(n).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d })
export const pct = (n: number | null | undefined, d = 1) => (n === null || n === undefined ? '—' : `${(n * 100).toFixed(d)}%`)
export const compact = (n: number) => Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(n)
