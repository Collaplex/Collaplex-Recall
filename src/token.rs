use std::collections::HashMap;

/// 极简切词：ASCII 按词，CJK 按双字，其他字母按词。
/// 中文不用分词器也能得到可用的 BM25 词项；单字词项的命中面太大
/// （「多」能命中 4.6 万段），所以只给孤立单字兜底。
pub fn terms(s: &str) -> Vec<String> {
    fn flush_word(word: &mut String, out: &mut Vec<String>) {
        if !word.is_empty() {
            out.push(std::mem::take(word));
        }
    }
    fn flush_run(run: &mut Vec<char>, out: &mut Vec<String>) {
        if run.is_empty() {
            return;
        }
        if run.len() == 1 {
            out.push(run[0].to_string());
        }
        for w in run.windows(2) {
            out.push(w.iter().collect());
        }
        run.clear();
    }

    let mut out = Vec::new();
    let mut word = String::new();
    let mut run: Vec<char> = Vec::new();

    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            flush_run(&mut run, &mut out);
            word.push(c.to_ascii_lowercase());
        } else if is_cjk(c) {
            flush_word(&mut word, &mut out);
            run.push(c);
        } else if c.is_alphabetic() {
            flush_run(&mut run, &mut out);
            word.push(c.to_ascii_lowercase());
        } else {
            flush_word(&mut word, &mut out);
            flush_run(&mut run, &mut out);
        }
    }
    flush_word(&mut word, &mut out);
    flush_run(&mut run, &mut out);
    out
}

pub fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3400..=0x4DBF   // 扩展 A
        | 0x4E00..=0x9FFF // 基本区
        | 0xF900..=0xFAFF // 兼容汉字
        | 0x20000..=0x2FA1F // 扩展 B–F
    )
}

/// 粗估 token：中文按 1 字 1 token，其余按 4 字符 1 token。
/// 只用于预算截断，不追求和具体 tokenizer 完全一致。
pub fn estimate_tokens(s: &str) -> usize {
    let mut cjk = 0usize;
    let mut other = 0usize;
    for c in s.chars() {
        if is_cjk(c) {
            cjk += 1;
        } else {
            other += 1;
        }
    }
    cjk + other.div_ceil(4)
}

/// 按 token 预算截断，UTF-8 安全；超出时以省略号结尾。
pub fn truncate_tokens(s: &str, max_tokens: usize) -> String {
    let mut out = String::new();
    let mut used = 0.0f64;
    let limit = max_tokens.max(1) as f64;
    for c in s.chars() {
        used += if is_cjk(c) { 1.0 } else { 0.25 };
        if used > limit {
            out.push('…');
            return out;
        }
        out.push(c);
    }
    out
}

/// 返回 (词项, 词频) 与文档总词数（含重复），供 BM25 建索引。
pub fn term_freqs(s: &str) -> (Vec<(String, i32)>, i32) {
    let mut map: HashMap<String, i32> = HashMap::new();
    let mut total = 0i32;
    for t in terms(s) {
        *map.entry(t).or_default() += 1;
        total += 1;
    }
    let mut v: Vec<(String, i32)> = map.into_iter().collect();
    v.sort();
    (v, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_emits_bigrams() {
        let t = terms("校园网");
        assert_eq!(t, vec!["校园", "园网"]);
    }

    #[test]
    fn single_cjk_char_is_kept() {
        assert_eq!(terms("鱼"), vec!["鱼"]);
    }

    #[test]
    fn token_estimate_and_truncation() {
        assert_eq!(estimate_tokens("校园网"), 3);
        assert_eq!(estimate_tokens("abcd"), 1);
        let t = truncate_tokens("一二三四五六", 3);
        assert!(t.starts_with("一二三"));
        assert!(t.ends_with('…'));
    }

    #[test]
    fn ascii_words_are_lowercased() {
        let t = terms("WAN 会话 TTL");
        assert!(t.contains(&"wan".to_string()));
        assert!(t.contains(&"ttl".to_string()));
        assert!(t.contains(&"会话".to_string()));
    }
}
