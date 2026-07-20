//! gate_bridge_resp — 对【桥回译给客户端的响应】(bridge_resp / phase*_bridge_resp) 的 usage,
//! 跑 agent-comm 已有的 #17 face_purity 门(check.rs:301),测桥后账面纯度。
//!
//! 与 gate_scan 的区别:gate_scan 对 face2_resp(上游真实响应,face=openai)跑;
//! 本工具对 bridge_resp(CCswitch 回译给客户端的 Responses 响应,face=responses)跑。
//! 二者是【桥前】与【桥后】两个不同的计费视图。
//!
//! ★ 诚实边界:#17 只测 usage 字段【多出】(impurity),测不到字段【丢失】(桥前有桥后无)。
//! "缓存计费字段丢失"需要桥前桥后对比,SPEC 目前无对应门,不在本工具范围。
//!
//! 零新增判断逻辑:只做 读文件 → 提取 model/usage → 调 check_face_purity → 打印原始返回。
//!
//! 用法: cargo run -p agent-comm --example gate_bridge_resp -- <scenarios根> [face=responses]

use agent_comm::check::{check_face_purity, check_model_identity, Finding};
use agent_comm::ResponseEnvelope;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn main() {
    let root = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: gate_bridge_resp <scenarios根> [face=responses]");
            std::process::exit(2);
        }
    };
    let face = std::env::args().nth(2).unwrap_or_else(|| "responses".to_string());

    let mut files = Vec::new();
    walk_bridge_resp(Path::new(&root), &mut files);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  gate_bridge_resp · 桥后回译响应 usage · #17 face_purity 门     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("root  : {root}");
    println!("face  : {face}  (桥后回译给客户端的响应协议)");
    println!("files : {} 个 bridge_resp", files.len());
    println!();

    let mut impurity = 0usize;
    let mut bill_pollute = 0usize;
    let mut reroute = 0usize;
    let mut no_usage = 0usize;
    let mut parse_err = 0usize;

    for f in &files {
        let cell = f.strip_prefix(&root).unwrap_or(f).display().to_string();
        let body = match load_body(f) {
            Some(b) => b,
            None => {
                parse_err += 1;
                println!("[PARSE_ERR] {cell}");
                continue;
            }
        };
        let usage = extract_u64_usage(&body);
        if usage.is_empty() {
            no_usage += 1;
            continue;
        }
        let env = ResponseEnvelope {
            echoed_model: body.get("model").and_then(|v| v.as_str()).map(String::from),
            usage,
            stop_reason: None,
            response_fingerprint: None,
        };
        let findings = check_face_purity(&face, &env);
        if findings.is_empty() {
            continue;
        }
        println!("[FINDING] {cell}");
        for fd in &findings {
            match fd {
                Finding::FaceImpurity { leaked_fields, .. } => {
                    impurity += 1;
                    println!("    #17 FaceImpurity(behav): leaked={leaked_fields:?}");
                }
                Finding::ForeignUsageField { polluted_fields, .. } => {
                    bill_pollute += 1;
                    println!("    #17 ForeignUsageField(bill): polluted={polluted_fields:?}");
                }
                Finding::Reroute { .. } => {
                    reroute += 1;
                    println!("    #16 Reroute: {fd:?}");
                }
            }
        }
    }

    println!();
    println!("── 总计(论文 #17 门原始判定) ──");
    println!("  bridge_resp 扫描 : {}", files.len());
    println!("  无 usage 字段    : {no_usage}");
    println!("  parse 错误       : {parse_err}");
    println!("  #17 FaceImpurity : {impurity}");
    println!("  #17 bill_pollute : {bill_pollute}");
    println!("  #16 reroute      : {reroute}");
    let _ = check_model_identity; // model 身份门在 FaceImpurity 分支同源,此处 usage 面为主
}

fn walk_bridge_resp(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk_bridge_resp(&p, out);
        } else if let Some(n) = p.file_name().and_then(|n| n.to_str()) {
            // bridge_resp.json / phase1_bridge_resp.json / phase2_bridge_resp.json /
            // second_bridge_resp.json — CCswitch 回译给客户端的响应
            if n.ends_with("bridge_resp.json") {
                out.push(p);
            }
        }
    }
}

fn load_body(path: &Path) -> Option<Value> {
    let raw = std::fs::read_to_string(path).ok()?;
    let outer: Value = serde_json::from_str(&raw).ok()?;
    let body = outer.get("body").cloned().unwrap_or(outer);
    match body {
        Value::String(s) => serde_json::from_str(&s).ok(),
        v => Some(v),
    }
}

fn extract_u64_usage(body: &Value) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    if let Some(usage) = body.get("usage").and_then(|v| v.as_object()) {
        for (k, v) in usage {
            if let Some(n) = v.as_u64() {
                out.insert(k.clone(), n);
            }
        }
    }
    out
}
