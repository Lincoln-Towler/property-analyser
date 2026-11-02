// Passes the suggestions on only if there are any, so the save step is skipped otherwise.
const { suggestions } = $('Prepare rows').first().json;
return suggestions.length ? [{ json: { rows: suggestions } }] : [];
