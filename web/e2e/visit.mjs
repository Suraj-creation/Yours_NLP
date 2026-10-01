// End-to-end check of every route: renders without console or page errors, no stuck
// loaders, no error boxes, no horizontal overflow on a phone. Screenshots for review.
//   node e2e/visit.mjs [baseUrl] [outDir] [--dark] [--mobile] [--only=/path]
import { chromium } from 'playwright'
import fs from 'node:fs'

const args = process.argv.slice(2)
const base = args.find((a) => a.startsWith('http')) ?? 'http://127.0.0.1:8000'
const out = args.find((a) => !a.startsWith('http') && !a.startsWith('--')) ?? 'e2e/shots'
const dark = args.includes('--dark'), mobile = args.includes('--mobile')
const only = args.find((a) => a.startsWith('--only='))?.slice(7)
const ROUTES = ['/', '/corpus', '/statistics', '/scale', '/tokenization', '/preprocessing', '/bpe', '/pos', '/custom-pos', '/ner',
  '/ngrams', '/index', '/search', '/pipelines', '/evaluation', '/justification', '/evidence', '/downloads']
fs.mkdirSync(out, { recursive: true })

const pre = '/opt/pw-browsers/chromium'  // preinstalled Chromium when the pinned build is absent
const browser = await chromium.launch(fs.existsSync(pre) ? { executablePath: fs.statSync(pre).isDirectory() ? undefined : pre } : {})
const ctx = await browser.newContext({
  viewport: mobile ? { width: 390, height: 844 } : { width: 1440, height: 900 }, deviceScaleFactor: 1,
  colorScheme: dark ? 'dark' : 'light',
})
let failures = 0
for (const r of only ? [only] : ROUTES) {
  const page = await ctx.newPage()
  const errors = []
  page.on('console', (m) => { if (m.type() === 'error') errors.push(`console: ${m.text()}`) })
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`))
  page.on('response', (res) => { if (res.status() >= 400) errors.push(`http ${res.status()} ${res.url()}`) })
  const t0 = Date.now()
  await page.goto(base + r, { waitUntil: 'networkidle', timeout: 180000 })
  // wait until no loader is visible (live endpoints may take a while on first call)
  await page.waitForFunction(() => !document.body.innerText.match(/\bLoading\b|Searching|Training…|Building or loading/), null, { timeout: 180000 }).catch(() => errors.push('loader still visible'))
  await page.waitForTimeout(900)
  const text = await page.evaluate(() => document.body.innerText)
  const m = text.match(/\b(4\d\d|5\d\d): .{0,80}/)
  if (m) errors.push(`error box: ${m[0]}`)
  if (text.includes('Page not found')) errors.push('route not found')
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
  if (overflow > 2) errors.push(`horizontal overflow ${overflow}px`)
  const charts = await page.locator('canvas').count()
  const name = (r === '/' ? 'home' : r.slice(1)) + (dark ? '-dark' : '') + (mobile ? '-mobile' : '')
  await page.screenshot({ path: `${out}/${name}.png`, fullPage: true })
  const ms = Date.now() - t0
  console.log(`${errors.length ? 'FAIL' : 'ok  '} ${r.padEnd(15)} ${String(ms).padStart(6)}ms  canvases=${charts}${errors.length ? '\n     ' + errors.slice(0, 6).join('\n     ') : ''}`)
  if (errors.length) failures++
  await page.close()
}
await browser.close()
console.log(failures ? `${failures} route(s) failed` : 'all routes passed')
process.exit(failures ? 1 : 0)
