// Your locations, and how SQM Research identifies each one.
// Capital cities use a region code; towns use their postcode.
// To add a location: copy a line, set the name exactly as it should appear on your
// site, and change the postcode.
const LOCATIONS = [
  { location: 'Sydney NSW',       sqm: 'region=nsw-Sydney&type=c' },
  { location: 'Brisbane QLD',     sqm: 'region=qld-Brisbane&type=c' },
  { location: 'Perth WA',         sqm: 'region=wa-Perth&type=c' },
  { location: 'Adelaide SA',      sqm: 'region=sa-Adelaide&type=c' },
  { location: 'Tamworth NSW',     sqm: 'postcode=2340' },
  { location: 'Newcastle NSW',    sqm: 'postcode=2300' },
  { location: 'Albany WA',        sqm: 'postcode=6330' },
  { location: 'Geelong VIC',      sqm: 'postcode=3220' },
  { location: 'Sale VIC',         sqm: 'postcode=3850' },
  { location: 'Port-Adelaide SA', sqm: 'postcode=5015' },
  { location: 'Wodonga VIC',      sqm: 'postcode=3690' },
];

// Three SQM pages per location: vacancy (monthly), asking rents and asking prices (weekly)
const PAGES = ['vacancy-rates', 'weekly-rents', 'asking-property-prices'];

return LOCATIONS.flatMap(l => PAGES.map(page => ({
  json: { location: l.location, page, url: `https://sqmresearch.com.au/property/${page}?${l.sqm}&t=1` },
})));
