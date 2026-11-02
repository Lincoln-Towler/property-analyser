"""Builds the 'Location Data (SQM)' n8n workflow from the tested Code-node scripts."""
import json
import pathlib
import uuid

HERE = pathlib.Path(__file__).parent
code = lambda f: (HERE / "code" / f).read_text().strip()
SUPABASE_CRED = {"supabaseApi": {"id": "0NLZ4l4z7Vzt7w2t", "name": "Supabase account"}}
nid = lambda: str(uuid.uuid4())

nodes = [
    {
        "parameters": {"rule": {"interval": [{"field": "weeks", "triggerAtDay": [1], "triggerAtHour": 7, "triggerAtMinute": 15}]}},
        "type": "n8n-nodes-base.scheduleTrigger", "typeVersion": 1.3, "position": [-600, 300],
        "id": nid(), "name": "Every Monday 7:15am",
    },
    {
        "parameters": {"jsCode": code("locations_list.js")},
        "type": "n8n-nodes-base.code", "typeVersion": 2, "position": [-380, 300],
        "id": nid(), "name": "Locations",
    },
    {
        "parameters": {
            "url": "={{ $json.url }}",
            "options": {
                "response": {"response": {"responseFormat": "text"}},
                # one page every 1.5 seconds, to be gentle on SQM's site
                "batching": {"batch": {"batchSize": 1, "batchInterval": 1500}},
            },
        },
        "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.4, "position": [-160, 300],
        "id": nid(), "name": "Fetch SQM pages",
        "onError": "continueRegularOutput", "retryOnFail": True, "maxTries": 2, "waitBetweenTries": 5000,
    },
    {
        "parameters": {"jsCode": code("location_metrics.js")},
        "type": "n8n-nodes-base.code", "typeVersion": 2, "position": [60, 300],
        "id": nid(), "name": "Calculate location metrics",
    },
    {
        "parameters": {
            "method": "POST",
            "url": "https://uorrpukzqwyvzbewgudk.supabase.co/rest/v1/property_data?on_conflict=date,location,metric_name",
            "authentication": "predefinedCredentialType", "nodeCredentialType": "supabaseApi",
            "sendHeaders": True,
            "headerParameters": {"parameters": [{"name": "Prefer", "value": "resolution=merge-duplicates,return=minimal"}]},
            "sendBody": True, "specifyBody": "json",
            "jsonBody": "={{ JSON.stringify($json.rows) }}",
            "options": {},
        },
        "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.4, "position": [280, 300],
        "id": nid(), "name": "Save to property_data", "credentials": SUPABASE_CRED,
    },
    {
        "parameters": {
            "content": "## Location data from SQM Research\n"
                       "Every Monday: fetches vacancy, asking rents and asking prices for each location, "
                       "works out **vacancy, rental yield, 12-month growth and asking price**, and saves them to `property_data`.\n\n"
                       "- Days on market and sales volume are left alone (manual entries).\n"
                       "- To add a location, edit the list in the **Locations** step.\n"
                       "- A page that fails just skips that location's figure; see `skipped` in the Calculate step.",
            "height": 260, "width": 460, "color": 4,
        },
        "type": "n8n-nodes-base.stickyNote", "typeVersion": 1, "position": [-640, -10],
        "id": nid(), "name": "About this workflow",
    },
]

chain = ["Every Monday 7:15am", "Locations", "Fetch SQM pages", "Calculate location metrics", "Save to property_data"]
connections = {a: {"main": [[{"node": b, "type": "main", "index": 0}]]} for a, b in zip(chain, chain[1:])}

workflow = {
    "name": "Location Data (SQM)",
    "nodes": nodes,
    "connections": connections,
    "pinData": {},
    "settings": {"executionOrder": "v1"},
}
out = HERE.parent / "workflows" / "Location Data (SQM).json"
out.write_text(json.dumps(workflow, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {out.name}: {len(nodes)} nodes")
