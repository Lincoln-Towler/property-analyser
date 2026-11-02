// Picks videos that still need analysing (max 3 per run, oldest first) and attaches the
// current flags, so Claude can see them.
// Failed videos are retried up to 3 times.
const videos = $('Parse feed').first().json.videos;
const seen = new Map($('Get analysed videos').all()
  .map(i => i.json)
  .filter(r => r.video_id)
  .map(r => [r.video_id, r]));

const flags = $('Get current flags').all().map(i => i.json).filter(f => f.flag_key);
if (!flags.length) throw new Error('Could not load btc_context_flags');

const todo = videos
  .filter(v => {
    const prev = seen.get(v.video_id);
    return !prev || (prev.status !== 'analysed' && (prev.attempts || 0) < 3);
  })
  .sort((a, b) => a.published_at.localeCompare(b.published_at))
  .slice(0, 3)
  .map(v => ({ json: { ...v, attempts: ((seen.get(v.video_id) || {}).attempts || 0) + 1, flags } }));

return todo; // nothing new -> the workflow stops here quietly
