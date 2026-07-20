//! gate_scan — 批量把已捕获的 face2(实际发给上游的报文) 喂给 agent-comm **已有的、
//! 论文已证明的**检测函数,不新增任何判定逻辑。
//!
//! 本文件只做三件事:
//!   1. 递归找 `*_face2_req.json` / 同目录同前缀的 `*_face2_resp.json` 配对(纯 I/O 扫描)
//!   2. 把 req.body 喂给 `split_envelope(face, native)`(论文 §已有函数),
//!      再把返回的 conv 喂给 check.rs 已有的结构门:
//!        - `find_orphan_toolresults`  (#20 悬空 tool_result)
//!        - `find_abandoned_toolcalls` (#19 被弃 tool_call)
//!      把 resp.body 的 model/usage 字段(纯字段提取,零判断)组装成 `ResponseEnvelope`,
//!      喂给已有的响应侧门:
//!        - `check_model_identity` (#16 reroute)
//!        - `check_face_purity`    (#17 usage 字段污染,即计费门)
//!   3. 打印每个门的原始返回值(Debug repr),不发明任何"保真/丢失"标签 ——
//!      判定标准 100% 在 check.rs 里,这里只转发。
//!
//! 用法:
//!   cargo run -p agent-comm --example gate_scan -- <root_dir> [face=openai]
//!   root_dir: 捕获数据根目录(如 CCswitch 的 captures/scenarios 或 captures/behavior)
//!   face:     face2 一侧的协议族,默认 openai(CCswitch/LiteLLM 都是 openai chat;
//!             MoonBridge 一侧应传 anthropic)
//!
//! 隐私:只打印字段名/id/计数,不打印任何正文原文。

use agent_comm::check::{
    check_face_purity, check_model_identity, find_abandoned_toolcalls, find_orphan_toolresults,
};
use agent_comm::{split_envelope, ResponseEnvelope};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn main() {
    let mut args = std::env::args().skip(1);
    let root = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: gate_scan <root_dir> [face=openai]");
            std::process::exit(2);
        }
    };
    let face = args.next().unwrap_or_else(|| "openai".to_string());

    let mut pairs = Vec::new();
    walk_find_req_resp_pairs(Path::new(&root), &mut pairs);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  gate_scan · agent-comm 既有检测门(check.rs) · 批量扫描          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("root  : {root}");
    println!("face  : {face}");
    println!("cells : {} 个 face2 req/resp 配对", pairs.len());
    println!();

    let mut n_orphan = 0usize;
    let mut n_abandoned = 0usize;
    let mut n_reroute = 0usize;
    let mut n_impurity = 0usize;
    let mut n_bill_pollute = 0usize;
    let mut n_up_loss = 0usize;
    let mut n_split_err = 0usize;
    let mut n_parse_err = 0usize;

    for (req_path, resp_path) in &pairs {
        let cell = req_path
            .strip_prefix(&root)
            .unwrap_or(req_path)
            .display()
            .to_string();

        let req_body = match load_body(req_path) {
            Some(b) => b,
            None => {
                n_parse_err += 1;
                println!("[PARSE_ERR] {cell} — face2_req 读取/解析失败");
                continue;
            }
        };
        let resp_body = match load_body(resp_path) {
            Some(b) => b,
            None => {
                n_parse_err += 1;
                println!("[PARSE_ERR] {cell} — face2_resp 读取/解析失败");
                continue;
            }
        };

        // ── ①③ 请求侧: split_envelope → conv,喂结构门(#19/#20) ──
        let (conv, _env, up_loss) = match split_envelope(&face, &req_body) {
            Ok(t) => t,
            Err(e) => {
                n_split_err += 1;
                println!("[SPLIT_ENVELOPE_ERR] {cell} — {e}");
                continue;
            }
        };
        n_up_loss += up_loss.len();

        let orphans = find_orphan_toolresults(&conv);
        let abandoned = find_abandoned_toolcalls(&conv);
        n_orphan += orphans.len();
        n_abandoned += abandoned.len();

        // ── ② 响应侧: 纯字段提取(model/usage) → ResponseEnvelope,喂 #16/#17 门 ──
        let requested_model = req_body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let echoed_model = resp_body
            .get("model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let usage = extract_u64_usage(&resp_body);

        let resp_env = ResponseEnvelope {
            echoed_model,
            usage,
            stop_reason: None,
            response_fingerprint: None,
        };

        let reroute = check_model_identity(&requested_model, &resp_env);
        let purity_findings = check_face_purity(&face, &resp_env);
        if reroute.is_some() {
            n_reroute += 1;
        }
        for f in &purity_findings {
            match f {
                agent_comm::check::Finding::FaceImpurity { .. } => n_impurity += 1,
                agent_comm::check::Finding::ForeignUsageField { .. } => n_bill_pollute += 1,
                _ => {}
            }
        }

        // ── 只在有发现时打印该 cell(安静通过的不刷屏,末尾有总计) ──
        let has_finding = !orphans.is_empty()
            || !abandoned.is_empty()
            || !up_loss.is_empty()
            || reroute.is_some()
            || !purity_findings.is_empty();
        if has_finding {
            println!("[FINDING] {cell}");
            for l in &up_loss {
                println!(
                    "    up_loss: [{}] turn={} recoverable={} — {}",
                    l.dropped_kind, l.turn_index, l.recoverable, l.note
                );
            }
            for o in &orphans {
                println!("    #20 orphan_toolresult: {o:?}");
            }
            for a in &abandoned {
                println!("    #19 abandoned_toolcall: {a:?}");
            }
            if let Some(r) = &reroute {
                println!("    #16 reroute: {r:?}");
            }
            for f in &purity_findings {
                println!("    #17 face_purity: {f:?}");
            }
        }
    }

    println!();
    println!("── 总计(论文既有门的原始判定,零自造标签) ──");
    println!("  cells 扫描           : {}", pairs.len());
    println!("  parse/split 错误      : parse={n_parse_err} split={n_split_err}");
    println!("  up_loss(codec级)      : {n_up_loss}");
    println!("  #20 orphan_toolresult : {n_orphan}");
    println!("  #19 abandoned_toolcall: {n_abandoned}");
    println!("  #16 reroute           : {n_reroute}");
    println!("  #17 face_impurity     : {n_impurity}");
    println!("  #17 bill_pollute      : {n_bill_pollute}");
}

/// 递归找 `*_face2_req.json`,同目录下把文件名的 `_req.json` 换成 `_resp.json` 作为配对。
/// 找不到配对的 req 跳过(不算错误 —— 有些捕获可能响应写失败,如实不计入)。
fn walk_find_req_resp_pairs(dir: &Path, out: &mut Vec<(PathBuf, PathBuf)>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_find_req_resp_pairs(&path, out);
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.ends_with("_face2_req.json") {
                let resp_name = name.replace("_face2_req.json", "_face2_resp.json");
                let resp_path = path.with_file_name(resp_name);
                if resp_path.exists() {
                    out.push((path.clone(), resp_path));
                }
            }
        }
    }
}

/// 读取一个 face2_req/resp.json,取出 `body` 字段(捕获格式统一是 `{path, body}`)。
/// body 若是字符串,先按整段 JSON 解析;若失败再按 SSE(`data: {...}` 逐行) 兼容解析 ——
/// 纯粹是把同一份数据的两种序列化形式(整段 JSON / SSE 分块)都能读出来,不涉及任何判断。
fn load_body(path: &Path) -> Option<Value> {
    let raw = std::fs::read_to_string(path).ok()?;
    let outer: Value = serde_json::from_str(&raw).ok()?;
    let body = outer.get("body").cloned().unwrap_or(outer);
    match body {
        Value::String(s) => serde_json::from_str(&s).ok().or_else(|| parse_sse(&s)),
        v => Some(v),
    }
}

/// 把 `data: {json}\n\n` 形式的 SSE 流重建成单个 JSON 对象:model 取第一个能解出的分块的值,
/// usage 取【最后一个含非空 usage 的分块】(OpenAI 兼容流式响应的通行做法:usage 只出现在
/// 结束前的那个分块)。这是格式重建,不是判断 —— 与本项目此前 Python 端 parse_sse 同一做法。
fn parse_sse(raw: &str) -> Option<Value> {
    let mut model: Option<Value> = None;
    let mut usage: Option<Value> = None;
    let mut any = false;
    for line in raw.lines() {
        let line = line.trim();
        let Some(payload) = line.strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        any = true;
        if model.is_none() {
            model = chunk.get("model").cloned();
        }
        if let Some(u) = chunk.get("usage") {
            if !u.is_null() {
                usage = Some(u.clone());
            }
        }
    }
    if !any {
        return None;
    }
    let mut obj = serde_json::Map::new();
    if let Some(m) = model {
        obj.insert("model".to_string(), m);
    }
    if let Some(u) = usage {
        obj.insert("usage".to_string(), u);
    }
    Some(Value::Object(obj))
}

/// 从响应 body 的顶层 `usage` 对象里,取出所有【顶层、标量、非负整数】字段。
/// 嵌套对象(如 `prompt_tokens_details`)不展开 —— 这不是判断,是 `ResponseEnvelope.usage:
/// BTreeMap<String,u64>` 这个类型本身的形状要求(#17 门比较的就是顶层 key 集合)。
fn extract_u64_usage(resp_body: &Value) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    if let Some(usage) = resp_body.get("usage").and_then(|v| v.as_object()) {
        for (k, v) in usage {
            if let Some(n) = v.as_u64() {
                out.insert(k.clone(), n);
            }
        }
    }
    out
}
