//! v2-client --url URL --engine NAME --model NAME --prompts FILE --out FILE [--n-gen 128] [--rounds 3]
//!
//! The same client as `_v2_http.py`, line for line, for a machine without Python: the Anthropic
//! Messages API, streamed, greedy, one request at a time; the five passages of each length, a
//! warm-up discarded, `rounds` rounds. TTFT = first non-empty content_block_delta - request sent;
//! prefill = input_tokens (message_start) / TTFT; decode = (output_tokens - 1) / (last delta - first
//! delta), output_tokens from message_delta.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::time::Instant;

struct Shot { ttft: f64, first: f64, last: f64, inp: Option<u64>, out: Option<u64> }

fn one(url: &str, engine: &str, model: &str, text: &str, n_gen: u64) -> Result<Option<Shot>, String> {
    let mut body = json!({"model": model, "max_tokens": n_gen, "temperature": 0.0, "stream": true,
        "messages": [{"role": "user", "content": text}]});
    if engine.starts_with("llama") {
        body["cache_prompt"] = json!(false);
    }
    let t0 = Instant::now();
    let resp = ureq::post(url).set("anthropic-version", "2023-06-01").set("content-type", "application/json").send_string(&body.to_string()).map_err(|e| e.to_string())?;
    let reader = BufReader::new(resp.into_reader());
    let (mut first, mut last, mut inp, mut out) = (None::<f64>, 0.0, None, None);
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let Some(data) = line.strip_prefix("data:") else { continue };
        let now = t0.elapsed().as_secs_f64();
        let d: Value = serde_json::from_str(data.trim()).map_err(|e| e.to_string())?;
        match d["type"].as_str() {
            Some("message_start") => inp = d["message"]["usage"]["input_tokens"].as_u64(),
            Some("content_block_delta") => {
                let delta = &d["delta"];
                let has = delta["text"].as_str().map_or(false, |s| !s.is_empty()) || delta["thinking"].as_str().map_or(false, |s| !s.is_empty());
                if has {
                    first.get_or_insert(now);
                    last = now;
                }
            }
            Some("message_delta") => {
                out = d["usage"]["output_tokens"].as_u64();
                if inp.is_none() {
                    inp = d["usage"]["input_tokens"].as_u64();
                }
            }
            Some("error") => return Err(d.to_string()),
            _ => {}
        }
    }
    Ok(first.map(|f| Shot { ttft: f, first: f, last, inp, out }))
}

fn median(v: &mut Vec<f64>) -> Option<f64> {
    if v.is_empty() { return None; }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    Some(if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 })
}

fn spread(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let mut s = v.to_vec();
    let m = median(&mut s).unwrap();
    (s[s.len() - 1] - s[0]) / m * 100.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str, d: Option<&str>| -> String {
        args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned().or(d.map(String::from)).unwrap_or_else(|| panic!("{k} is required"))
    };
    let (url, engine, model) = (get("--url", None), get("--engine", None), get("--model", None));
    let (prompts, out_file) = (get("--prompts", None), get("--out", None));
    let n_gen: u64 = get("--n-gen", Some("128")).parse().unwrap();
    let rounds: usize = get("--rounds", Some("3")).parse().unwrap();
    let p: Value = serde_json::from_str(&std::fs::read_to_string(&prompts).unwrap()).unwrap();
    let mut keys: Vec<&String> = p.as_object().unwrap().keys().collect();
    keys.sort_by_key(|k| k.parse::<u64>().unwrap());
    let mut rows = Vec::new();
    for k in keys {
        let passages: Vec<&str> = p[k].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let _ = one(&url, &engine, &model, passages[0], 8); // warm-up, discarded
        let (mut ttft, mut pf, mut dec) = (Vec::new(), Vec::new(), Vec::new());
        let (mut inps, mut outs) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
        for _ in 0..rounds {
            for text in &passages {
                let s = match one(&url, &engine, &model, text, n_gen) {
                    Ok(Some(s)) => s,
                    Ok(None) => continue,
                    Err(e) => { eprintln!("request error: {e}"); continue; }
                };
                let Some(inp) = s.inp else { continue };
                ttft.push(s.ttft * 1000.0);
                pf.push(inp as f64 / s.ttft);
                inps.insert(inp);
                if let Some(o) = s.out {
                    outs.insert(o);
                    if o > 1 && s.last > s.first {
                        dec.push((o - 1) as f64 / (s.last - s.first));
                    }
                }
            }
        }
        let (tm, pm, dm) = (median(&mut ttft.clone()), median(&mut pf.clone()), median(&mut dec.clone()));
        println!("{engine} {model} p{k:>5}: in {:?} · TTFT {:.2} ms · prefill {:.0} tok/s ({:.1} %) · decode {:.2} tok/s ({:.1} %) · out {:?} · n={}",
            inps, tm.unwrap_or(f64::NAN), pm.unwrap_or(f64::NAN), spread(&pf), dm.unwrap_or(f64::NAN), spread(&dec), outs, ttft.len());
        rows.push(json!({"engine": engine, "model": model, "prompt_key": k.parse::<u64>().unwrap(), "samples": ttft.len(),
            "input_tokens": inps, "output_tokens": outs, "ttft_ms_median": tm, "ttft_ms_all": ttft, "prefill_tps_median": pm,
            "prefill_spread_pct": spread(&pf), "decode_tps_median": dm, "decode_spread_pct": spread(&dec), "decode_tps_all": dec}));
    }
    std::fs::write(&out_file, serde_json::to_string_pretty(&rows).unwrap()).unwrap();
}
