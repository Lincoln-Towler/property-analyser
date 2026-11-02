// Turns Claude's answers into rows for btc_video_analysis and btc_flag_suggestions.
// Suggestions are checked before saving: unknown flags are dropped, duplicates of existing
// flags become updates, and modifiers are kept between -0.5 and +0.5.
const built = $('Build Claude request').all();
const CATEGORIES = new Set(['CYCLE', 'MACRO', 'ON_CHAIN']);
const clamp = n => Math.max(-0.5, Math.min(0.5, Math.round(Number(n) * 10) / 10));
const slug = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 80);
const text = (s, max) => (s && String(s).trim()) ? String(s).trim().slice(0, max) : null;

const videos = [];
const suggestions = [];

$input.all().forEach((item, i) => {
  const b = built[i].json;
  const row = {
    video_id: b.video_id, title: b.title, published_at: b.published_at, url: b.url,
    transcript_chars: b.transcript_chars || null, attempts: b.attempts || 1,
    summary: null, stance: null, key_levels: null, model: null, status: 'failed', error: null,
  };

  try {
    if (b.error) throw new Error(b.error);
    const r = item.json;
    if (r.error || !Array.isArray(r.content)) throw new Error('Claude request failed: ' + JSON.stringify(r.error || r).slice(0, 300));
    if (r.stop_reason !== 'end_turn') throw new Error('Claude stopped early: ' + r.stop_reason);
    const answer = JSON.parse(r.content.find(c => c.type === 'text').text);

    row.summary = text(answer.summary, 1200);
    row.stance = ['bullish', 'bearish', 'neutral', 'mixed'].includes(answer.stance) ? answer.stance : null;
    row.key_levels = text(answer.key_levels, 500);
    row.model = r.model || null;
    row.status = 'analysed';

    const known = new Map(b.flags.map(f => [f.flag_key, f]));
    for (const s of (answer.suggestions || []).slice(0, 8)) {
      const flag_key = slug(s.flag_key);
      let action = s.action;
      if (!flag_key || !['add', 'update', 'retire'].includes(action)) continue;
      if (action === 'add' && known.has(flag_key)) action = 'update';
      if (action !== 'add' && !known.has(flag_key)) continue;             // can't change a flag that doesn't exist
      if (action === 'retire' && known.get(flag_key).is_active === false) continue; // already retired
      if (suggestions.some(x => x.video_id === b.video_id && x.flag_key === flag_key)) continue; // one change per flag per video

      suggestions.push({
        video_id: b.video_id,
        action,
        flag_key,
        flag_label: text(s.flag_label, 200),
        category: CATEGORIES.has(s.category) ? s.category : null,
        current_value: text(s.current_value, 500),
        score_modifier: action === 'retire' || !Number.isFinite(Number(s.score_modifier)) ? null : clamp(s.score_modifier),
        is_active: action === 'retire' ? false : action === 'add' ? true : (typeof s.is_active === 'boolean' ? s.is_active : null),
        reason: text(s.reason, 500),
      });
    }
  } catch (e) {
    row.error = String(e.message || e).slice(0, 500);
  }
  videos.push(row);
});

// If several videos in this run suggest a change to the same flag, keep only the newest video's
// (videos are processed oldest first, so later entries win).
const newest = new Map();
for (const s of suggestions) newest.set(s.flag_key, s);

return [{ json: { videos, suggestions: [...newest.values()] } }];
