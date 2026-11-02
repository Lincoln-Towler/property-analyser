# Property Analyser

A personal dashboard for the Australian property market, with a separate Bitcoin cycle signal. It's one person's research tool, and nothing here is financial advice.

**Live site:** https://property-analyser-three.vercel.app

It runs by itself. n8n workflows collect the data into Supabase, and a read-only Next.js site displays it. There's no login, and nothing on the site can change the data.

| Folder | What it is |
|---|---|
| `web/` | The site: Next.js on Vercel. See [`web/README.md`](web/README.md). |
| `n8n/` | The data workflows: source, tests and importable JSON. See [`n8n/README.md`](n8n/README.md). |
| `supabase/migrations/` | The database schema and a record of every production change. |
| `obsidian-vault/` | Notes: architecture, changelog and handoffs. |
| `legacy/` | The retired Streamlit app, kept as the reference for the scoring tests. |

To work on it, start with [`CLAUDE.md`](CLAUDE.md) and the newest handoff note in `obsidian-vault/`.
