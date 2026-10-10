//! Scene settings from the page URL (`?n=1000000&work=heavy`) in the browser,
//! or from `key=value` / `--key=value` arguments natively.

use std::collections::HashMap;
use std::str::FromStr;

/// Every setting given, by name.
pub fn args() -> HashMap<String, String> {
    let mut out = HashMap::new();
    #[cfg(not(target_arch = "wasm32"))]
    for arg in std::env::args().skip(1) {
        let arg = arg.trim_start_matches("--");
        if let Some((k, v)) = arg.split_once('=') {
            out.insert(k.to_string(), v.to_string());
        } else {
            out.insert(arg.to_string(), String::new());
        }
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(search) = web_sys::window().and_then(|w| w.location().search().ok()) {
        for pair in search
            .trim_start_matches('?')
            .split('&')
            .filter(|p| !p.is_empty())
        {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            out.insert(k.to_string(), v.to_string());
        }
    }
    out
}

/// The setting `name` parsed as `T`, or `default` when it's missing or invalid.
pub fn arg<T: FromStr>(name: &str, default: T) -> T {
    args()
        .get(name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
