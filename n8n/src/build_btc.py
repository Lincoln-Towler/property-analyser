"""Builds the 'Cowen Videos -> BTC Flag Suggestions' n8n workflow from the tested Code-node scripts."""
import json
import pathlib
import uuid

HERE = pathlib.Path(__file__).parent
code = lambda f: (HERE / "code" / f).read_text().strip()
nid = lambda: str(uuid.uuid4())
SUPABASE = "https://uorrpukzqwyvzbewgudk.supabase.co/rest/v1"
SUPABASE_CRED = {"supabaseApi": {"id": "0NLZ4l4z7Vzt7w2t", "name": "Supabase account"}}

def http(name, pos, url, method="GET", fmt="json", extra=None, **settings):
    # fmt=None leaves n8n on Autodetect, which copes with Supabase's empty reply after a save
    params = {"url": url, "options": {"response": {"response": {"responseFormat": fmt}}} if fmt else {}}
    if method != "GET":
        params["method"] = method
    params.update(extra or {})
    return {"parameters": params, "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.4,
            "position": pos, "id": nid(), "name": name, **settings}

def supabase(name, pos, path, method="GET", body_expr=None, prefer=None, **settings):
    extra = {"authentication": "predefinedCredentialType", "nodeCredentialType": "supabaseApi"}
    if prefer:
        extra.update({"sendHeaders": True, "headerParameters": {"parameters": [{"name": "Prefer", "value": prefer}]}})
    if body_expr:
        extra.update({"sendBody": True, "specifyBody": "json", "jsonBody": body_expr})
    node = http(name, pos, f"{SUPABASE}/{path}", method, fmt=None if method == "POST" else "json", extra=extra, **settings)
    node["credentials"] = SUPABASE_CRED
    return node

def code_node(name, pos, file):
    return {"parameters": {"jsCode": code(file)}, "type": "n8n-nodes-base.code", "typeVersion": 2,
            "position": pos, "id": nid(), "name": name}

RETRY = {"retryOnFail": True, "maxTries": 2, "waitBetweenTries": 5000}
y1, y2 = 200, 440
nodes = [
    {"parameters": {"rule": {"interval": [{"field": "days", "triggerAtHour": 8, "triggerAtMinute": 30}]}},
     "type": "n8n-nodes-base.scheduleTrigger", "typeVersion": 1.3, "position": [-900, y1], "id": nid(), "name": "Every day 8:30am"},
    http("Get Cowen's YouTube feed", [-680, y1], "https://www.youtube.com/feeds/videos.xml?channel_id=UCRvqjQPSeaWn-uEx-w0XOIg", fmt="text", **RETRY),
    code_node("Parse feed", [-460, y1], "btc_parse_feed.js"),
    supabase("Get analysed videos", [-240, y1], "btc_video_analysis?select=video_id,status,attempts&order=created_at.desc&limit=200",
             alwaysOutputData=True),
    supabase("Get current flags", [-20, y1],
             "btc_context_flags?select=flag_key,flag_label,category,current_value,score_modifier,is_active,updated_at&order=category.asc,flag_key.asc",
             executeOnce=True),
    supabase("Get pending suggestions", [200, y1], "btc_flag_suggestions?status=eq.pending&select=flag_key,action,video_id",
             executeOnce=True, alwaysOutputData=True),
    code_node("Keep new videos", [420, y1], "btc_keep_new.js"),
    # --- second row: per-video work ---
    http("Get transcript (Supadata)", [-680, y2], "https://api.supadata.ai/v1/transcript",
         extra={"authentication": "genericCredentialType", "genericAuthType": "httpHeaderAuth",
                "sendQuery": True, "queryParameters": {"parameters": [
                    {"name": "url", "value": "={{ $json.url }}"}, {"name": "lang", "value": "en"},
                    {"name": "text", "value": "true"}, {"name": "mode", "value": "native"}]}},
         onError="continueRegularOutput", **RETRY),
    code_node("Build Claude request", [-460, y2], "btc_build_request.js"),
    http("Ask Claude", [-240, y2], "https://api.anthropic.com/v1/messages", method="POST",
         extra={"authentication": "predefinedCredentialType", "nodeCredentialType": "anthropicApi",
                "sendHeaders": True, "headerParameters": {"parameters": [{"name": "anthropic-version", "value": "2023-06-01"}]},
                "sendBody": True, "specifyBody": "json", "jsonBody": "={{ JSON.stringify($json.body) }}"},
         onError="continueRegularOutput", retryOnFail=True, maxTries=2, waitBetweenTries=10000),
    code_node("Prepare rows", [-20, y2], "btc_prepare_rows.js"),
    supabase("Save videos", [200, y2], "btc_video_analysis?on_conflict=video_id", method="POST",
             body_expr="={{ JSON.stringify($json.videos) }}", prefer="resolution=merge-duplicates,return=minimal"),
    code_node("Suggestions to save", [420, y2], "btc_suggestions_to_save.js"),
    supabase("Save suggestions", [640, y2], "btc_flag_suggestions", method="POST",
             body_expr="={{ JSON.stringify($json.rows) }}", prefer="return=minimal"),
    {"parameters": {"content":
        "## Cowen videos → BTC flag suggestions\n"
        "Every morning: finds new Benjamin Cowen videos (official channel only, no Shorts), gets the transcript from Supadata, "
        "and asks Claude Sonnet which `btc_context_flags` to **update, retire or add**.\n\n"
        "Nothing changes your score until you approve it: in Supabase open **btc_flag_suggestions** and set `status` to "
        "**approved** (applies it) or **rejected**. Only the newest suggestion per flag stays pending.\n\n"
        "Score rules (in Supabase): flags move the score at most ±2 points; max 20 active flags.\n\n"
        "Max 3 videos per run. A video that fails is retried up to 3 days.",
        "height": 300, "width": 520, "color": 4},
     "type": "n8n-nodes-base.stickyNote", "typeVersion": 1, "position": [-1460, 120], "id": nid(), "name": "About this workflow"},
]

# One video at a time: Supadata's free plan rejects requests sent together (HTTP 429).
# Ask Claude can take a minute or two per video.
claude = next(n for n in nodes if n["name"] == "Ask Claude")["parameters"]["options"]
claude.update({"timeout": 300000, "batching": {"batch": {"batchSize": 1, "batchInterval": 1000}}})
transcript = next(n for n in nodes if n["name"] == "Get transcript (Supadata)")["parameters"]["options"]
transcript.update({"timeout": 120000, "batching": {"batch": {"batchSize": 1, "batchInterval": 3000}}})

chain = ["Every day 8:30am", "Get Cowen's YouTube feed", "Parse feed", "Get analysed videos", "Get current flags",
         "Get pending suggestions", "Keep new videos", "Get transcript (Supadata)", "Build Claude request", "Ask Claude", "Prepare rows",
         "Save videos", "Suggestions to save", "Save suggestions"]
connections = {a: {"main": [[{"node": b, "type": "main", "index": 0}]]} for a, b in zip(chain, chain[1:])}

workflow = {"name": "Cowen Videos → BTC Flag Suggestions", "nodes": nodes, "connections": connections,
            "pinData": {}, "settings": {"executionOrder": "v1"}}
out = HERE.parent / "workflows" / "Cowen Videos - BTC Flag Suggestions.json"
out.write_text(json.dumps(workflow, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {out.name}: {len(nodes)} nodes")
