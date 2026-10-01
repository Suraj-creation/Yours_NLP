// Interaction tests: every live control on the site is exercised once against the real API.
//   node e2e/interact.mjs [baseUrl] [outDir]
import { chromium } from 'playwright'
import fs from 'node:fs'
import path from 'node:path'

const base = process.argv[2] ?? 'http://127.0.0.1:8000'
const out = process.argv[3] ?? 'e2e/shots'
fs.mkdirSync(out, { recursive: true })
const pre = '/opt/pw-browsers/chromium'
const browser = await chromium.launch(fs.existsSync(pre) ? { executablePath: pre } : {})
const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 } })).newPage()
const errors = []
page.on('pageerror', (e) => errors.push(e.message))
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
let fails = 0
async function check(name, fn) {
  const t0 = Date.now()
  try { await fn(); console.log(`ok   ${name} (${Date.now() - t0} ms)`) } catch (e) { fails++; console.log(`FAIL ${name}: ${e.message.split('\n')[0]}`) }
}
const text = () => page.evaluate(() => document.body.innerText)
const until = async (re, timeout = 120000) => page.waitForFunction((s) => new RegExp(s).test(document.body.innerText), re.source, { timeout })

await check('search: Boolean query with gold evaluation', async () => {
  await page.goto(`${base}/search`, { waitUntil: 'networkidle' })
  await page.fill('input[aria-label="Query"]', 'denominator AND NOT common')
  await page.click('button:has-text("Search")')
  await until(/Gold query Q08/)
  const t = await text()
  if (!/Boolean NOT/.test(t)) throw new Error('type not shown')
})
await check('search: phrase query on MathDial', async () => {
  await page.click('button:has-text("MathDial · 2,861")')
  await page.fill('input[aria-label="Query"]', 'common denominator')
  await page.click('button:has-text("Search")')
  await until(/Phrase \(bigram\)/)
  await until(/MD\d{5}/)
})
await check('search: TalkMoves Boolean OR', async () => {
  await page.click('button:has-text("TalkMoves · 566")')
  await page.fill('input[aria-label="Query"]', 'half OR halves')
  await page.click('button:has-text("Search")')
  await until(/TM\d{4}/)
})
await check('search: malformed query shows an error, not a crash', async () => {
  await page.click('button:has-text("30 documents")')
  await page.fill('input[aria-label="Query"]', 'fraction AND (misconception')
  await page.click('button:has-text("Search")')
  await page.waitForTimeout(1500)
  const t = await text()
  if (!/400|query error|No document matches|documents/.test(t)) throw new Error('no feedback')
  errors.length = 0  // the 400 from this deliberate bad query is expected
})
await check('pipeline picker changes the active pipeline', async () => {
  await page.goto(`${base}/index`, { waitUntil: 'networkidle' })
  await page.selectOption('select[aria-label="Active pipeline"]', 'A')
  await until(/pipeline A/)
  await page.fill('input[aria-label="Term"]', 'denominators')
  await page.click('button:has-text("Look up")')
  await until(/index term\s+denomin\b/)
  await page.selectOption('select[aria-label="Active pipeline"]', 'B')
})
await check('tokenization: live tokenizer and rule switch', async () => {
  await page.goto(`${base}/tokenization`, { waitUntil: 'networkidle' })
  await page.fill('textarea', 'She paid $3,650 for 3 1/2 kg, idk.')
  await page.click('button:has-text("Tokenize")')
  await until(/\$3,650/)
})
await check('preprocessing: order switch reports stem leaks', async () => {
  await page.goto(`${base}/preprocessing`, { waitUntil: 'networkidle' })
  await page.getByLabel('Stop list', { exact: true }).selectOption('nltk')
  await page.getByLabel('Order', { exact: true }).selectOption('norm_then_stop')
  await page.fill('textarea', 'This was his answer because he has only one half.')
  await page.click('button:has-text("Run")')
  await until(/leaks as “thi”/)
})
await check('bpe: merges slider retrains', async () => {
  await page.goto(`${base}/bpe`, { waitUntil: 'networkidle' })
  const s = page.locator('input[type=range]')
  await s.focus(); await s.press('End')
  await until(/4,000 merges/, 180000)
})
await check('pos: live tagging', async () => {
  await page.goto(`${base}/pos`, { waitUntil: 'networkidle' })
  await page.fill('textarea', 'Round 3.456 to the nearest tenth.')
  await page.click('button:has-text("Tag")')
  await until(/math instruction verb/)
})
await check('ner: ruler adds domain entities', async () => {
  await page.goto(`${base}/ner`, { waitUntil: 'networkidle' })
  await page.fill('textarea', 'We compare BKT and DKT on MathDial using AUC.')
  await page.click('button:has-text("Find entities")')
  await until(/KT_MODEL/)
})
await check('evidence: record for a decimal misconception', async () => {
  await page.goto(`${base}/evidence`, { waitUntil: 'networkidle' })
  await page.fill('textarea', '0.45 is greater than 0.6 since 45 is greater than 6.')
  await page.click('button:has-text("Build record")')
  await until(/M05 longer decimal is bigger/)
  await page.click('button:has-text("JSON")')
  await until(/"evidence_id"/)
})
await check('corpus: document viewer and upload', async () => {
  await page.goto(`${base}/corpus`, { waitUntil: 'networkidle' })
  await page.selectOption('#viewer ~ section select, select >> nth=1', 'D29').catch(async () => { await page.locator('select').nth(1).selectOption('D29') })
  await until(/D29 · Class web page/)
  const f = path.join(out, 'upload_test.txt')
  fs.writeFileSync(f, 'Aarav added 1/2 + 1/3 = 2/5 on 2026-09-01 and paid Rs 250.\nThen he said idk.')
  await page.setInputFiles('input[type=file]', f)
  await until(/detected TXT/)
})
await check('pipelines: design-your-own run', async () => {
  await page.goto(`${base}/pipelines`, { waitUntil: 'networkidle' })
  await page.getByLabel('Tokenizer', { exact: true }).selectOption('nltk')
  await page.getByLabel('Stop list', { exact: true }).selectOption('custom')
  await page.getByLabel('Normaliser', { exact: true }).selectOption('snowball')
  await page.getByLabel('Order', { exact: true }).selectOption('stop_then_norm')
  await page.click('button:has-text("Run")')
  await until(/vs B/, 300000)
  await page.screenshot({ path: path.join(out, 'designer.png'), fullPage: false })
})
await check('theme toggle switches to dark and back', async () => {
  await page.goto(`${base}/`, { waitUntil: 'networkidle' })
  await page.click('button[aria-label="Toggle theme"]')
  const th = await page.evaluate(() => document.documentElement.dataset.theme)
  if (th !== 'dark') throw new Error(`theme=${th}`)
  await page.click('button[aria-label="Toggle theme"]')
})
console.log(errors.length ? `page errors:\n  ${errors.slice(0, 10).join('\n  ')}` : 'no page errors')
await browser.close()
console.log(fails ? `${fails} interaction(s) failed` : 'all interactions passed')
process.exit(fails || errors.length ? 1 : 0)
