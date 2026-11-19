use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    /// 笔记内序号，从 1 开始；重新切块时整篇替换。
    pub ord: i32,
    /// 标题路径，例如 "校园网认证排查 › 掉线规律"。
    pub heading: String,
    pub text: String,
    pub hash: String,
}

pub fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}

struct Unit {
    heading: String,
    text: String,
}

fn heading_path(stack: &[(usize, String)]) -> String {
    stack
        .iter()
        .map(|(_, t)| t.as_str())
        .collect::<Vec<_>>()
        .join(" › ")
}

/// 按标题 + 空行切出段落，再把同标题下的段落打包到 max_chars。
/// 超长段落按字符数硬切（UTF-8 安全）。v0 不做重叠，见 README。
pub fn chunk_markdown(src: &str, max_chars: usize) -> Vec<Chunk> {
    let max_chars = max_chars.max(64);
    let units = parse_units(src);

    let mut chunks = Vec::new();
    let mut ord = 0i32;
    let mut buf = String::new();
    let mut buf_heading = String::new();

    for unit in units {
        if unit.heading != buf_heading && !buf.trim().is_empty() {
            push_chunk(&mut chunks, &mut ord, &buf_heading, &buf);
            buf.clear();
        }
        buf_heading = unit.heading.clone();

        if char_len(&unit.text) > max_chars {
            if !buf.trim().is_empty() {
                push_chunk(&mut chunks, &mut ord, &buf_heading, &buf);
                buf.clear();
            }
            for piece in split_by_chars(&unit.text, max_chars) {
                push_chunk(&mut chunks, &mut ord, &unit.heading, &piece);
            }
            continue;
        }

        if !buf.is_empty() {
            buf.push_str("\n\n");
        }
        buf.push_str(&unit.text);

        if char_len(&buf) >= max_chars {
            push_chunk(&mut chunks, &mut ord, &buf_heading, &buf);
            buf.clear();
        }
    }

    if !buf.trim().is_empty() {
        push_chunk(&mut chunks, &mut ord, &buf_heading, &buf);
    }

    chunks
}

/// 第一行 `# 标题` 作为笔记标题；没有就用文件名。
pub fn title_of(src: &str, fallback: &str) -> String {
    for line in src.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("# ") {
            let rest = rest.trim();
            if !rest.is_empty() {
                return rest.to_string();
            }
        }
        if !trimmed.is_empty() {
            break;
        }
    }
    fallback.to_string()
}

fn push_chunk(chunks: &mut Vec<Chunk>, ord: &mut i32, heading: &str, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    *ord += 1;
    chunks.push(Chunk {
        ord: *ord,
        heading: heading.to_string(),
        text: text.to_string(),
        hash: sha256_hex(&format!("{heading}\u{1f}{text}")),
    });
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn split_by_chars(text: &str, max: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    chars
        .chunks(max)
        .map(|c| c.iter().collect::<String>())
        .collect()
}

fn parse_units(src: &str) -> Vec<Unit> {
    let mut units = Vec::new();
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut para = String::new();
    let mut in_code = false;

    let flush = |para: &mut String, units: &mut Vec<Unit>, stack: &[(usize, String)]| {
        let text = para.trim();
        if text.is_empty() {
            para.clear();
            return;
        }
        units.push(Unit {
            heading: heading_path(stack),
            text: text.to_string(),
        });
        para.clear();
    };

    for line in src.lines() {
        let trimmed = line.trim_start();

        if trimmed.starts_with("```") {
            in_code = !in_code;
            if !para.is_empty() {
                para.push('\n');
            }
            para.push_str(line);
            continue;
        }

        if !in_code {
            let level = trimmed.chars().take_while(|c| *c == '#').count();
            if (1..=6).contains(&level) {
                let rest = trimmed[level..].trim();
                if !rest.is_empty() {
                    flush(&mut para, &mut units, &stack);
                    while stack.last().map(|(l, _)| *l >= level).unwrap_or(false) {
                        stack.pop();
                    }
                    stack.push((level, rest.to_string()));
                    continue;
                }
            }
        }

        if line.trim().is_empty() && !in_code {
            flush(&mut para, &mut units, &stack);
        } else {
            if !para.is_empty() {
                para.push('\n');
            }
            para.push_str(line);
        }
    }

    flush(&mut para, &mut units, &stack);
    units
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_by_heading_and_packs() {
        let src = "# 一\n\n段落A。\n\n段落B。\n\n## 一.1\n\n段落C。\n";
        let chunks = chunk_markdown(src, 64);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading, "一");
        assert!(chunks[0].text.contains("段落A"));
        assert!(chunks[0].text.contains("段落B"));
        assert_eq!(chunks[1].heading, "一 › 一.1");
        assert!(chunks[1].text.contains("段落C"));
    }

    #[test]
    fn splits_long_paragraph() {
        let src = "# 长\n\n".to_string() + &"字".repeat(200);
        let chunks = chunk_markdown(&src, 64);
        assert!(chunks.len() >= 4);
        assert!(chunks.iter().all(|c| char_len(&c.text) <= 64));
    }

    #[test]
    fn ignores_headings_inside_code_fence() {
        let src = "# 一\n\n```\n# 不是标题\n```\n\n正文。\n";
        let chunks = chunk_markdown(src, 256);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("# 不是标题"));
        assert_eq!(chunks[0].heading, "一");
    }
}
