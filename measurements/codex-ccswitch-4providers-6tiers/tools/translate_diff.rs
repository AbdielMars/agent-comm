//! translate_diff — 用 agent-comm 已有的 `translate()` 生成"理论正确的参照译文",
//! 与桥(CCswitch/LiteLLM/...)实际产出的 face2 做结构对比,把 9 族损失分类学真正跑出来。
//!
//! 为什么需要这个(而不是只跑 gate_scan 的 4 个响应侧/结构门):
//!   gate_scan 只对【已经翻译完的 face2】单独 up(),这只能查 face2 自身是否自洽
//!   (悬空 tool_result / usage 纯度),查不出"桥在翻译过程中丢了什么" ——
//!   因为丢没丢,face2 单独看不出来,必须跟 face1(原始语义)对照。
//!
//! 方法(零新增判断逻辑,全部调用已有 pub fn):
//!   1. translate("responses", "openai", face1_body)                    ← 已有函数
//!      → (predicted_face2, translate_loss)  translate_loss 就是【协议固有/不可避免】
//!        的 9 族损失清单(任何忠实的桥都躲不掉的那部分,不是 CCswitch 的锅)
//!   2. split_envelope("openai", predicted_face2) → conv_predicted        ← 已有函数
//!      split_envelope("openai", actual_face2)    → conv_actual           ← 已有函数
//!      ★ 两侧都用【同一个 openai codec】lift,不再是 responses vs openai 的跨
//!        codec 比较,STEP2 发现的 reasoning 陷阱在这里不成立(见 §0 说明)
//!   3. conv_predicted.normalize() == conv_actual.normalize()?            ← 已有方法
//!      不等 → 逐 turn 比较 Content 构成(纯计数/纯字段读取,非判断) → 报告差异
//!
//! §0 唯一需要人工解读的地方(如实标注,不是判断逻辑里的分支):
//!   agent-comm 的 responses codec 把 face1 里任何 `type=="reasoning"` 条目都判定为
//!   "server-managed/encrypted,不可信地表达"(见 STEP2 调研),所以 translate() 产出的
//!   predicted_face2 里 reasoning 永远是空的 —— 这是 codec 的既定假设,不是这次新加的判断。
//!   因此当差异只发生在 Thinking 内容上时,该差异不能简单读成"CCswitch 丢了东西",
//!   而应读成"agent-comm 的参照译文在此处比 CCswitch 更保守"——这条 caveat 照实打印,
//!   不是我临时决定的,是 STEP2 已经查明的 codec 已知假设。
//!
//! 用法:
//!   cargo run -p agent-comm --example translate_diff -- <scenarios根目录>
//!
//! 隐私:只打印结构计数与 id,不打印任何正文原文。

use agent_comm::{split_envelope, translate, Content};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn main() {
    let root = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: translate_diff <scenarios根目录>");
            std::process::exit(2);
        }
    };

    let pairs = collect_pairs(Path::new(&root));

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  translate_diff · translate()参照译文 vs 桥实际译文 · 9族对照   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("root  : {root}");
    println!("pairs : {} 个(face1↔face2)可比对单元", pairs.len());
    println!();

    let mut equal = 0usize;
    let mut differ = 0usize;
    let mut reasoning_only = 0usize;
    let mut err = 0usize;
    // scenario → dropped_kind 计数(translate 自带的协议固有损失)
    let mut inherent: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    // scenario → 差异类别计数(实际比参照多/少的结构差异,按 content 类型分桶)
    let mut extra: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();

    for cell in &pairs {
        let scenario = cell.scenario.clone();

        let face1 = match load_json(&cell.face1) {
            Some(v) => v,
            None => {
                err += 1;
                println!("[ERR] {} — face1 读取失败: {:?}", cell.label, cell.face1);
                continue;
            }
        };
        let actual_face2 = match load_body_maybe_sse(&cell.face2_req) {
            Some(v) => v,
            None => {
                err += 1;
                println!(
                    "[ERR] {} — face2_req 读取失败: {:?}",
                    cell.label, cell.face2_req
                );
                continue;
            }
        };

        // ── ①translate() 生成协议固有参照 + 固有损失清单 ──
        let (predicted_face2, translate_loss) = match translate("responses", "openai", &face1) {
            Ok(t) => t,
            Err(e) => {
                err += 1;
                println!("[ERR] {} — translate() 失败: {e}", cell.label);
                continue;
            }
        };
        for l in &translate_loss {
            *inherent
                .entry(scenario.clone())
                .or_default()
                .entry(l.dropped_kind.clone())
                .or_insert(0) += 1;
        }

        // ── ②③ 同一 openai codec 双侧 lift,比对 ──
        let (conv_pred, _e1, _u1) = match split_envelope("openai", &predicted_face2) {
            Ok(t) => t,
            Err(e) => {
                err += 1;
                println!("[ERR] {} — predicted lift 失败: {e}", cell.label);
                continue;
            }
        };
        let (conv_act, _e2, _u2) = match split_envelope("openai", &actual_face2) {
            Ok(t) => t,
            Err(e) => {
                err += 1;
                println!("[ERR] {} — actual lift 失败: {e}", cell.label);
                continue;
            }
        };
        let np = conv_pred.normalize();
        let na = conv_act.normalize();

        if np == na {
            equal += 1;
            continue;
        }

        // ── 不等: 逐turn结构差异(纯计数,非判断) ──
        let (buckets, thinking_only) = structural_diff(&np, &na);
        if thinking_only {
            reasoning_only += 1;
        } else {
            differ += 1;
        }
        for (k, n) in &buckets {
            *extra
                .entry(scenario.clone())
                .or_default()
                .entry(k.clone())
                .or_insert(0) += n;
        }

        println!(
            "[DIFFER{}] {}",
            if thinking_only { "·仅thinking,见§0 caveat" } else { "" },
            cell.label
        );
        for (k, n) in &buckets {
            println!("    {k}: predicted与actual相差 {n} 处");
        }
        if !thinking_only {
            println!("    predicted 序列: {}", shape_seq(&np));
            println!("    actual    序列: {}", shape_seq(&na));
        }
    }

    println!();
    println!("── 总计 ──");
    println!("  可比对单元: {}", pairs.len());
    println!("  完全相等(predicted == actual): {equal}");
    println!("  仅thinking差异(§0 caveat,不算CCswitch的锅): {reasoning_only}");
    println!("  非thinking的真实结构差异: {differ}");
    println!("  错误: {err}");
    println!();
    println!("── translate()自带的协议固有损失(9族,任何忠实桥都躲不掉) 按场景×族 ──");
    for (s, m) in &inherent {
        println!("  {s}: {m:?}");
    }
    println!();
    println!("── predicted vs actual 的结构差异 按场景×类别(CCswitch自己引入的) ──");
    for (s, m) in &extra {
        println!("  {s}: {m:?}");
    }
}

struct Cell {
    label: String,
    scenario: String,
    face1: PathBuf,
    face2_req: PathBuf,
}

/// 按摸排确认的三种形态收集 (face1, face2_req) 配对:
///   phase1/phase2 两阶段 · bridge_in 单阶段 · S19 无face1(跳过,不适用)
fn collect_pairs(scenarios_root: &Path) -> Vec<Cell> {
    let mut out = Vec::new();
    let Ok(scenario_dirs) = std::fs::read_dir(scenarios_root) else {
        return out;
    };
    for sd in scenario_dirs.flatten() {
        if !sd.path().is_dir() {
            continue;
        }
        let scenario = sd.file_name().to_string_lossy().to_string();
        let Ok(model_dirs) = std::fs::read_dir(sd.path()) else {
            continue;
        };
        for md in model_dirs.flatten() {
            let mdir = md.path();
            if !mdir.is_dir() {
                continue;
            }
            let model = md.file_name().to_string_lossy().to_string();
            // 两阶段
            for phase in ["phase1", "phase2"] {
                let f1 = mdir.join(format!("face1_{phase}.json"));
                let f2dir = mdir.join(format!("face2_{phase}"));
                if f1.exists() && f2dir.is_dir() {
                    if let Some(req) = first_req_in(&f2dir) {
                        out.push(Cell {
                            label: format!("{scenario}/{model}/{phase}"),
                            scenario: scenario.clone(),
                            face1: f1,
                            face2_req: req,
                        });
                    }
                }
            }
            // 单阶段
            let f1 = mdir.join("face1_bridge_in.json");
            let f2dir = mdir.join("face2");
            if f1.exists() && f2dir.is_dir() {
                if let Some(req) = first_req_in(&f2dir) {
                    out.push(Cell {
                        label: format!("{scenario}/{model}"),
                        scenario: scenario.clone(),
                        face1: f1,
                        face2_req: req,
                    });
                }
            }
            // S19(无face1) 等其它形态: 没有 face1 就没法做参照对比,如实不纳入,不报错
        }
    }
    out
}

fn first_req_in(dir: &Path) -> Option<PathBuf> {
    let mut best: Option<PathBuf> = None;
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if let Some(n) = p.file_name().and_then(|n| n.to_str()) {
            if n.ends_with("_face2_req.json") {
                if best.is_none() || Some(p.clone()) < best {
                    best = Some(p);
                }
            }
        }
    }
    best
}

fn load_json(path: &Path) -> Option<Value> {
    let raw = std::fs::read_to_string(path).ok()?;
    let outer: Value = serde_json::from_str(&raw).ok()?;
    // face1 捕获格式是 {"payload": {...}}
    Some(outer.get("payload").cloned().unwrap_or(outer))
}

fn load_body_maybe_sse(path: &Path) -> Option<Value> {
    let raw = std::fs::read_to_string(path).ok()?;
    let outer: Value = serde_json::from_str(&raw).ok()?;
    let body = outer.get("body").cloned().unwrap_or(outer);
    match body {
        Value::String(s) => serde_json::from_str(&s).ok(),
        v => Some(v),
    }
}

/// 逐turn把 Content 构成拆成计数桶,只做加减计数与 id 集合比较,不发明"严重度"标签。
/// 返回 (差异桶, 是否只有thinking差异)。
fn structural_diff(
    predicted: &agent_comm::Conversation,
    actual: &agent_comm::Conversation,
) -> (BTreeMap<String, usize>, bool) {
    let mut out = BTreeMap::new();
    let mut only_thinking = true;

    let pt = predicted.turns.len();
    let at = actual.turns.len();
    if pt != at {
        out.insert("turn_count".to_string(), (pt as i64 - at as i64).unsigned_abs() as usize);
        only_thinking = false;
    }

    for i in 0..pt.max(at) {
        let pcount = predicted.turns.get(i).map(census).unwrap_or_default();
        let acount = actual.turns.get(i).map(census).unwrap_or_default();
        for (k, pv) in &pcount {
            let av = acount.get(k).copied().unwrap_or(0);
            if *pv != av {
                let key = format!("turn{i}.{k}");
                out.insert(key, (*pv as i64 - av as i64).unsigned_abs() as usize);
                if k != "thinking" {
                    only_thinking = false;
                }
            }
        }
        for (k, av) in &acount {
            if !pcount.contains_key(k) {
                let key = format!("turn{i}.{k}");
                out.insert(key, *av);
                if k != "thinking" {
                    only_thinking = false;
                }
            }
        }
    }

    let has_diff = !out.is_empty();
    (out, only_thinking && has_diff)
}

/// 纯只读的结构形状打印:每个turn的role + 该turn里每个content的类型缩写,
/// 用逗号/竖线拼接。不打印任何正文,只打印类型名和角色名(纯字段读取,非判断)。
fn shape_seq(conv: &agent_comm::Conversation) -> String {
    conv.turns
        .iter()
        .map(|t| {
            let kinds: Vec<&str> = t
                .content
                .iter()
                .map(|c| match c {
                    Content::Text { .. } => "text",
                    Content::ToolCall { .. } => "toolcall",
                    Content::ToolResult { .. } => "toolresult",
                    Content::Thinking { .. } => "thinking",
                    Content::Media { .. } => "media",
                    Content::Video { .. } => "video",
                })
                .collect();
            format!("{:?}[{}]", t.role, kinds.join("+"))
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn census(turn: &agent_comm::Turn) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for c in &turn.content {
        let k = match c {
            Content::Text { .. } => "text",
            Content::ToolCall { .. } => "tool_call",
            Content::ToolResult { .. } => "tool_result",
            Content::Thinking { .. } => "thinking",
            Content::Media { .. } => "media",
            Content::Video { .. } => "video",
        };
        *m.entry(k.to_string()).or_insert(0) += 1;
    }
    m
}
