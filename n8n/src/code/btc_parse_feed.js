// Reads Benjamin Cowen's YouTube feed (his latest 15 uploads) and keeps recent full-length videos.
// Shorts are skipped. Videos must be at least 6 hours old so YouTube has made the captions.
const xml = String($input.first().json.data || '');
if (!xml.includes('<feed')) throw new Error('YouTube feed did not load');

const decode = s => s.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
  .replace(/&quot;/g, '"').replace(/&#39;/g, "'");
const HOUR = 3600000, DAY = 24 * HOUR, now = Date.now();

const videos = [...xml.matchAll(/<entry>([\s\S]*?)<\/entry>/g)]
  .map(m => {
    const get = re => (m[1].match(re) || [])[1];
    return {
      video_id: get(/<yt:videoId>([^<]+)<\/yt:videoId>/),
      title: decode(get(/<title>([^<]*)<\/title>/) || ''),
      published_at: get(/<published>([^<]+)<\/published>/),
      url: get(/<link rel="alternate" href="([^"]+)"/),
    };
  })
  .filter(v => v.video_id && v.url && !v.url.includes('/shorts/'))
  .filter(v => {
    const age = now - Date.parse(v.published_at);
    return age > 6 * HOUR && age < 7 * DAY;
  });

return [{ json: { videos } }];
