// Tests the Cowen transcript workflow's Code nodes with sample feed, transcript and Claude responses.
const fs = require('fs');
const path = require('path');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const read = f => fs.readFileSync(path.join(__dirname, 'code', f), 'utf8');
async function runNode(file, inputItems, nodes = {}) {
  const $input = { first: () => inputItems[0], all: () => inputItems };
  const $ = name => ({ all: () => nodes[name], first: () => nodes[name][0] });
  return new AsyncFunction('$input', '$', read(file))($input, $);
}
let failures = 0;
const check = (name, ok, detail = '') => { if (!ok) failures++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  -> ' + detail}`); };
const ago = h => new Date(Date.now() - h * 3600000).toISOString().replace('.000Z', '+00:00');

(async () => {
  // ---------- Parse feed ----------
  const entry = (id, title, hoursAgo, short = false) => `<entry><id>yt:video:${id}</id><yt:videoId>${id}</yt:videoId><yt:channelId>UCRvqjQPSeaWn-uEx-w0XOIg</yt:channelId><title>${title}</title><link rel="alternate" href="https://www.youtube.com/${short ? 'shorts/' + id : 'watch?v=' + id}"/><published>${ago(hoursAgo)}</published></entry>`;
  const xml = `<?xml version="1.0"?><feed xmlns:yt="http://www.youtube.com/xml/schemas/2015">` +
    entry('NEW1', 'Bitcoin: Too Fresh', 2) + entry('SHORT1', 'A Short', 20, true) +
    entry('VID_A', 'Bitcoin: Dooobious Spookulation', 30) + entry('VID_B', 'NFA Live! Bitcoin &amp; Fed Fallout', 80) +
    entry('VID_C', 'Gold: Dubious Speculation', 120) + entry('OLD1', 'Too Old', 24 * 9) + `</feed>`;
  const feed = (await runNode('btc_parse_feed.js', [{ json: { data: xml } }]))[0].json.videos;
  check('feed: skips Shorts, too-new (<6h) and old (>7 days)', JSON.stringify(feed.map(v => v.video_id)) === '["VID_A","VID_B","VID_C"]', JSON.stringify(feed.map(v => v.video_id)));
  check('feed: decodes &amp; in titles', feed[1].title === 'NFA Live! Bitcoin & Fed Fallout', feed[1].title);
  let threw = false; try { await runNode('btc_parse_feed.js', [{ json: { data: '<html>blocked</html>' } }]); } catch (e) { threw = true; }
  check('feed: stops with an error if YouTube feed fails', threw);

  // ---------- Keep new videos ----------
  const flags = [
    { flag_key: 'golden_cross_bull', flag_label: 'Golden cross', category: 'CYCLE', current_value: 'Sep 2026: crossed', score_modifier: 0.1, is_active: true, updated_at: '2026-09-18T00:00:00+00:00' },
    { flag_key: 'june_fed_boj_catalyst', flag_label: 'June Fed/BoJ window', category: 'MACRO', current_value: 'Jul 2026', score_modifier: -0.2, is_active: true, updated_at: '2026-07-31T00:00:00+00:00' },
    { flag_key: 'old_inactive', flag_label: 'Old', category: 'MACRO', current_value: '', score_modifier: -0.3, is_active: false, updated_at: '2026-05-26T00:00:00+00:00' },
  ];
  const analysed = [{ json: { video_id: 'VID_C', status: 'analysed', attempts: 1 } }, { json: { video_id: 'VID_B', status: 'failed', attempts: 1 } }];
  const flagItems = flags.map(f => ({ json: f }));
  const keep = await runNode('btc_keep_new.js', [{ json: {} }], { 'Parse feed': [{ json: { videos: feed } }], 'Get analysed videos': analysed, 'Get current flags': flagItems });
  check('keep: skips analysed, retries failed, oldest first', JSON.stringify(keep.map(k => k.json.video_id)) === '["VID_B","VID_A"]', JSON.stringify(keep.map(k => k.json.video_id)));
  check('keep: counts attempts', keep[0].json.attempts === 2 && keep[1].json.attempts === 1);
  const keep3 = await runNode('btc_keep_new.js', [{ json: {} }], { 'Parse feed': [{ json: { videos: feed } }], 'Get analysed videos': [{ json: { video_id: 'VID_B', status: 'failed', attempts: 3 } }], 'Get current flags': flagItems });
  check('keep: gives up after 3 failed attempts', !keep3.some(k => k.json.video_id === 'VID_B'));
  const none = await runNode('btc_keep_new.js', [{ json: {} }], { 'Parse feed': [{ json: { videos: [] } }], 'Get analysed videos': [{ json: {} }], 'Get current flags': flagItems });
  check('keep: attaches the current flags', keep[0].json.flags.length === 3 && keep[0].json.pending === undefined);
  check('keep: nothing new -> stops quietly', Array.isArray(none) && none.length === 0);

  // ---------- Build Claude request ----------
  const transcript = 'Bitcoin is trading around the 50 week moving average. '.repeat(40);
  const built = await runNode('btc_build_request.js', [{ json: { content: transcript, lang: 'en' } }, { json: { error: 'transcript-unavailable', message: 'No captions' } }], { 'Keep new videos': keep });
  const body = built[0].json.body;
  check('build: Sonnet 5.5 with JSON schema output', body.model === 'claude-sonnet-5-5' && body.output_config.format.type === 'json_schema' && body.output_config.format.schema.additionalProperties === false);
  check('build: lists flags and the active total', /golden_cross_bull \[CYCLE\] ACTIVE/.test(body.messages[0].content) && /2 active of max 20; active modifiers total -0\.1/.test(body.messages[0].content), body.messages[0].content.slice(0, 300));
  check('build: prompt states the cap and limit', /2 active of max 20/.test(body.messages[0].content) && /hard limit of 20 active flags/.test(body.system), body.messages[0].content.slice(0, 400));
  check('build: prompt says changes apply with no review, and nothing is left waiting', /apply automatically with no human review/.test(body.system) && !/waiting for review/.test(body.system + body.messages[0].content));
  check('build: transcript wrapped as data', /<transcript>\nBitcoin is trading/.test(body.messages[0].content));
  check('build: missing transcript becomes an error, not a request', !built[1].json.body && /No transcript/.test(built[1].json.error));
  const axios429 = { json: { error: { message: "Try spacing your requests out using the batching settings under 'Options'", name: 'AxiosError', stack: 'AxiosError: Request failed with status code 429\n    at settle (...)' } } };
  const built429 = await runNode('btc_build_request.js', [axios429], { 'Keep new videos': [keep[0]] });
  check('build: rate-limit error recorded briefly', built429[0].json.error === "No transcript: HTTP 429: Try spacing your requests out using the batching settings under 'Options'", built429[0].json.error);
  JSON.stringify(body); // must serialise cleanly

  // ---------- Prepare rows ----------
  const answer = {
    summary: 'Cowen argues the Q4 drop is following the midterm-year pattern.', stance: 'bearish', key_levels: 'Support ~$70K; resistance ~$85K',
    suggestions: [
      { action: 'retire', flag_key: 'june_fed_boj_catalyst', flag_label: '', category: 'MACRO', current_value: 'Nov 2026: window passed', score_modifier: -0.2, is_active: false, reason: 'June window is over' },
      { action: 'update', flag_key: 'golden_cross_bull', flag_label: 'Golden cross', category: 'CYCLE', current_value: 'Nov 2026: post-cross dip underway', score_modifier: 0.05, is_active: true, reason: 'Still valid' },
      { action: 'add', flag_key: 'Q4 Dump Underway', flag_label: 'Q4 dump underway', category: 'CYCLE', current_value: 'Nov 2026: Q4 weakness', score_modifier: -0.9, is_active: true, reason: 'New' },
      { action: 'add', flag_key: 'golden_cross_bull', flag_label: 'dup', category: 'CYCLE', current_value: 'x', score_modifier: 0.1, is_active: true, reason: 'dup of existing' },
      { action: 'update', flag_key: 'made_up_flag', flag_label: '', category: 'CYCLE', current_value: 'x', score_modifier: 0, is_active: true, reason: 'unknown' },
      { action: 'retire', flag_key: 'old_inactive', flag_label: '', category: 'MACRO', current_value: '', score_modifier: -0.3, is_active: false, reason: 'already off' },
    ],
  };
  const claudeOk = { json: { model: 'claude-sonnet-5-5', stop_reason: 'end_turn', content: [{ type: 'thinking', thinking: '...' }, { type: 'text', text: JSON.stringify(answer) }] } };
  const prepared = (await runNode('btc_prepare_rows.js', [claudeOk, { json: { error: { message: 'Bad request' } } }], { 'Build Claude request': built }))[0].json;
  const sug = prepared.suggestions;
  check('prepare: video saved as analysed with summary', prepared.videos[0].status === 'analysed' && prepared.videos[0].stance === 'bearish' && /Q4/.test(prepared.videos[0].summary));
  check('prepare: retire keeps is_active false, no modifier', sug.some(s => s.action === 'retire' && s.flag_key === 'june_fed_boj_catalyst' && s.is_active === false && s.score_modifier === null));
  check('prepare: modifier rounded to 0.1 and capped at ±0.5', sug.find(s => s.flag_key === 'golden_cross_bull' && s.action === 'update').score_modifier === 0.1 && sug.find(s => s.flag_key === 'q4_dump_underway').score_modifier === -0.5, JSON.stringify(sug));
  check('prepare: new flag key cleaned to snake_case', sug.some(s => s.action === 'add' && s.flag_key === 'q4_dump_underway'));
  check('prepare: retires are saved first, adds last', sug[0].action === 'retire' && sug[sug.length - 1].action === 'add', JSON.stringify(sug.map(s => s.action)));
  check('prepare: "add" of an existing flag becomes "update"', sug.filter(s => s.flag_key === 'golden_cross_bull').every(s => s.action === 'update'));
  check('prepare: unknown and already-retired flags dropped', !sug.some(s => s.flag_key === 'made_up_flag' || s.flag_key === 'old_inactive'));
  check('prepare: transcript failure recorded as failed (attempt counted)', prepared.videos[1].status === 'failed' && /No transcript/.test(prepared.videos[1].error) && prepared.videos[1].attempts === 1);
  const cutOff = { json: { stop_reason: 'max_tokens', content: [{ type: 'text', text: '{"summ' }] } };
  const p2 = (await runNode('btc_prepare_rows.js', [cutOff], { 'Build Claude request': [built[0]] }))[0].json;
  check('prepare: cut-off answer recorded as failed, no suggestions', p2.videos[0].status === 'failed' && /stopped early/.test(p2.videos[0].error) && p2.suggestions.length === 0);

  const ans2 = { summary: 'Newer view.', stance: 'mixed', key_levels: '', suggestions: [
    { action: 'update', flag_key: 'golden_cross_bull', flag_label: 'Golden cross', category: 'CYCLE', current_value: 'Nov 2026: newest view', score_modifier: 0.2, is_active: true, reason: 'newer' } ] };
  const claude2 = { json: { model: 'claude-sonnet-5-5', stop_reason: 'end_turn', content: [{ type: 'text', text: JSON.stringify(ans2) }] } };
  const twoVideos = await runNode('btc_prepare_rows.js', [claudeOk, claude2], { 'Build Claude request': [built[0], { json: { ...built[0].json, video_id: 'VID_A' } }] });
  const gc = twoVideos[0].json.suggestions.filter(s => s.flag_key === 'golden_cross_bull');
  check('prepare: same flag in two videos -> only the newer video kept', gc.length === 1 && gc[0].video_id === 'VID_A' && gc[0].current_value === 'Nov 2026: newest view', JSON.stringify(gc));

  // ---------- Suggestions to save ----------
  const s1 = await runNode('btc_suggestions_to_save.js', [{ json: {} }], { 'Prepare rows': [{ json: prepared }] });
  const s0 = await runNode('btc_suggestions_to_save.js', [{ json: {} }], { 'Prepare rows': [{ json: p2 }] });
  check('save step only runs when there are suggestions', s1.length === 1 && s1[0].json.rows.length === sug.length && s0.length === 0);

  console.log(`      sample suggestion row: ${JSON.stringify(sug[0])}`);
  console.log(failures ? `\n${failures} FAILED` : '\nAll tests passed');
  process.exit(failures ? 1 : 0);
})();
