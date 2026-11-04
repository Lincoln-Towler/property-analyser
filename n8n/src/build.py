"""Builds the fixed n8n workflow JSON from the original export plus the tested Code-node scripts."""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
CODE = HERE / "code"
original = json.loads((HERE / "original.json").read_text())
by_id = {n["id"]: n for n in original["nodes"]}

def code(name, with_abs_helpers=False):
    body = (CODE / name).read_text().strip()
    if with_abs_helpers:
        body = (CODE / "abs_helpers.js").read_text().strip() + "\n\n" + body
    return body

ABS = "https://data.api.abs.gov.au/rest/data/ABS,{flow}/{key}?lastNObservations={n}&dimensionAtObservation=AllDimensions&format=jsondata"

FETCH_SETTINGS = {"onError": "continueRegularOutput", "retryOnFail": True, "maxTries": 2, "waitBetweenTries": 5000}

def fetch(old_id, name, url, pos, fmt, never_error=False):
    response = {"responseFormat": fmt}
    if never_error:
        response["neverError"] = True
    return {
        "parameters": {"url": url, "options": {"response": {"response": response}}},
        "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.4,
        "position": pos, "id": old_id, "name": name, **FETCH_SETTINGS,
    }

def parse(old_id, name, js, pos):
    return {"parameters": {"jsCode": js}, "type": "n8n-nodes-base.code", "typeVersion": 2,
            "position": pos, "id": old_id, "name": name}

def save(old_id, name, pos):
    node = json.loads(json.dumps(by_id[old_id]))  # keep the working Supabase history upsert as-is
    node.update({"name": name, "position": pos, "onError": "continueRegularOutput"})
    return node

# (fetch_id, parse_id, save_id, label, fetch_name, url, response format, parse file, abs helpers?)
branches = [
    ("b1a934e0-6464-4f00-b8c3-836a37483adf", "1cd2b7aa-a56b-4d4d-a7e9-9ade6321fec4", "69d07c84-b719-4a5e-9a3b-eaec05be34be",
     "cash rate", "Fetch cash rate (RBA)", "https://www.rba.gov.au/statistics/cash-rate/", "text", "parse_cash_rate.js", False),
    None,  # credit growth has an extra step, built below
    ("2e022092-0e6d-4f64-b79e-221c71acddea", "74098f5c-75d9-45b0-87aa-252783da5e73", "2f9b283c-cc76-4f2f-92ed-653856264b76",
     "unemployment", "Fetch unemployment (ABS)", ABS.format(flow="LF", key="M13.3.1599.20.AUS.M", n=1), "json", "parse_unemployment.js", True),
    ("b8a7e142-4330-4b40-aeb6-d4d6ece78ad2", "1ac81767-6ddd-4998-b88d-3a9ac33ffebb", "792997c9-a4e2-4aa7-853a-90e14a963aaa",
     "building approvals", "Fetch building approvals (ABS)", ABS.format(flow="BA_GCCSA", key="1.1.9.TOT.TOT.10.AUS.M", n=1), "json", "parse_building_approvals.js", True),
    ("5070366a-5753-4b5a-be48-f600a33a412b", "8c3e0a6d-0f2e-4773-a1c5-0f2c347f2b1f", "dd62b473-ccf2-4050-aa11-83a92638f055",
     "wage growth", "Fetch wage growth (ABS)", ABS.format(flow="WPI", key="3.THRPEB.7.TOT.20.AUS.Q", n=1), "json", "parse_wage_growth.js", True),
    ("8619fb32-67ce-4d69-9919-aacba8508eb0", "1526449b-7869-424a-9bbd-5cf57953d760", "40501761-15c2-4bc7-9d38-83bfcf1cad4c",
     "population growth", "Fetch population (ABS)", ABS.format(flow="ERP_COMP_Q", key="10.AUS.Q", n=5), "json", "parse_population.js", True),
    ("672281f9-8215-4513-b7d8-cff3760cf7a7", "08c60cab-9d5a-42f9-b56b-4102d1136f53", "7118f9ba-2e63-494c-b429-6011729289d0",
     "clearance rate", "Fetch auction results", "https://propertyupdate.com.au/this-weekends-auction-results-what-happened-in-sydney-melbourne-brisbane-adelaide-canberra/", "text", "parse_clearance.js", False),
    ("c269d726-9a5f-4d7f-aea6-1402564f7b07", "f794b5ce-3209-477b-a2d5-d7e8dde7242d", "41f49731-4d42-4ed7-baf8-18b3378353f7",
     "vacancy rate", "Fetch vacancy rate (SQM)", "https://sqmresearch.com.au/property/vacancy-rates?national=1&t=1", "text", "parse_vacancy.js", False),
]

trigger = by_id["0f664fd8-565c-41b3-b37e-b40703d95c3d"]
trigger["position"] = [-1072, 672]
# Every morning, so a new figure is in within a day. All the saves are upserts, so a
# run that finds nothing new changes nothing, apart from the cash rate's daily row.
trigger["parameters"] = {"rule": {"interval": [{"field": "days", "triggerAtHour": 7}]}}
nodes = [trigger]
connections = {"Schedule Trigger": {"main": [[]]}}
starts = []

def link(a, b):
    connections.setdefault(a, {"main": [[]]})["main"][0].append({"node": b, "type": "main", "index": 0})

for row, branch in enumerate(branches):
    y = row * 192
    if branch is None:
        names = ["Build RBA credit URLs", "Fetch credit pages (RBA)", "Parse housing credit", "Save credit growth"]
        nodes += [
            parse("a1743a3b-eb67-458d-a6ec-acd54aeb022e", names[0], code("build_rba_urls.js"), [-848, y]),
            fetch("282a2cbd-4b73-49b3-b91d-20dd50050a88", names[1], "={{ $json.url }}", [-624, y], "text", never_error=True),
            parse("cbfd640a-71ab-4584-b436-26266f189a57", names[2], code("parse_housing_credit.js"), [-400, y]),
            save("a6154e16-f020-495b-ab37-95b1e3cba290", names[3], [-176, y]),
        ]
    else:
        f_id, p_id, s_id, label, f_name, url, fmt, js_file, helpers = branch
        names = [f_name, "Parse " + label, "Save " + label]
        nodes += [
            fetch(f_id, names[0], url, [-848, y], fmt),
            parse(p_id, names[1], code(js_file, helpers), [-624, y]),
            save(s_id, names[2], [-400, y]),
        ]
    starts.append(names[0])
    for a, b in zip(names, names[1:]):
        link(a, b)

for s in starts:
    link("Schedule Trigger", s)

nodes.append({
    "parameters": {
        "content": "## Economic indicators (fixed 2 Nov 2026)\n"
                   "Runs every morning at 7:00. Each row fetches one indicator, reads the number, and saves it to `economic_indicators_history`.\n\n"
                   "- If a source fails or looks wrong, that row saves nothing and the others keep running.\n"
                   "- The score audit shows any indicator that has gone stale.",
        "height": 220, "width": 420, "color": 4,
    },
    "type": "n8n-nodes-base.stickyNote", "typeVersion": 1,
    "position": [-1520, 520], "id": "5b0d7d2e-3c1a-4f8e-9a63-6f2a9c1e7b41", "name": "About this workflow",
})

fixed = {
    "name": original["name"],
    "nodes": nodes,
    "connections": connections,
    "pinData": {},
    "settings": original["settings"],
    "meta": original["meta"],
}
out = HERE.parent / "workflows" / "Economic Indicators.json"
out.write_text(json.dumps(fixed, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {out.name}: {len(nodes)} nodes (was {len(original['nodes'])})")
