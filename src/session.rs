use serde::Serialize;

/// 会话指针是二层表里唯一跨轮活着的东西：
/// 上轮命中 id、上轮问题、已解指代。它不落库，只随会话内存 + 每轮 JSONL 落盘。
#[derive(Debug, Clone, Serialize, Default)]
pub struct PointerHit {
    pub chunk_id: i64,
    pub path: String,
    pub heading: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedRef {
    pub mention: String,
    pub referent: String,
    pub chunk_id: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionPointer {
    pub session_id: String,
    pub turn: u32,
    pub last_query: Option<String>,
    pub last_hits: Vec<PointerHit>,
    pub resolved: Vec<ResolvedRef>,
}

impl SessionPointer {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            turn: 0,
            last_query: None,
            last_hits: Vec::new(),
            resolved: Vec::new(),
        }
    }

    /// 没有判题改写模型时，指针的确定性替身：
    /// 问题里出现指代词，就把上轮第一名当作指代对象。
    pub fn resolve(&self, question: &str) -> Option<ResolvedRef> {
        const ANAPHORA: &[&str] = &[
            "它",
            "这个",
            "那个",
            "这些",
            "那些",
            "上述",
            "上面",
            "刚才",
            "此条",
            "该笔记",
            "这篇",
        ];
        let top = self.last_hits.first()?;
        let mention = ANAPHORA.iter().find(|m| question.contains(**m))?;
        let referent = if top.heading.is_empty() {
            top.path.clone()
        } else {
            format!("{} › {}", top.path, top.heading)
        };
        Some(ResolvedRef {
            mention: (*mention).to_string(),
            referent,
            chunk_id: top.chunk_id,
        })
    }

    /// 检索用的问题：有指代时把指代对象拼进查询，只做确定性扩展。
    pub fn query_for(&self, question: &str) -> (String, Option<ResolvedRef>) {
        match self.resolve(question) {
            Some(r) => (format!("{question} {}", r.referent), Some(r)),
            None => (question.to_string(), None),
        }
    }

    /// 只记录本轮结果，不推进 turn（例如 /search）。
    pub fn remember(&mut self, query: &str, hits: &[(i64, String, String)]) {
        self.last_query = Some(query.to_string());
        self.last_hits = hits.iter().take(3).map(hit).collect();
    }

    /// 完整的一轮问答结束：推进 turn、记录命中、保存已解指代。
    pub fn advance(
        &mut self,
        query: &str,
        hits: &[(i64, String, String)],
        resolved: Option<ResolvedRef>,
    ) {
        self.turn += 1;
        self.remember(query, hits);
        if let Some(r) = resolved {
            self.resolved.push(r);
            if self.resolved.len() > 32 {
                self.resolved.remove(0);
            }
        }
    }
}

fn hit((chunk_id, path, heading): &(i64, String, String)) -> PointerHit {
    PointerHit {
        chunk_id: *chunk_id,
        path: path.clone(),
        heading: heading.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pointer_with_hit() -> SessionPointer {
        let mut p = SessionPointer::new("t".into());
        p.remember(
            "校园网为什么掉线",
            &[(7, "校园网.md".into(), "掉线规律".into())],
        );
        p
    }

    #[test]
    fn resolves_anaphora_to_top_hit() {
        let p = pointer_with_hit();
        let r = p.resolve("它一般持续多久？").unwrap();
        assert_eq!(r.mention, "它");
        assert!(r.referent.contains("掉线规律"));
        let (q, _) = p.query_for("它一般持续多久？");
        assert!(q.contains("校园网.md"));
    }

    #[test]
    fn no_anaphora_no_expansion() {
        let p = pointer_with_hit();
        let (q, r) = p.query_for("显存多大");
        assert_eq!(q, "显存多大");
        assert!(r.is_none());
    }

    #[test]
    fn advance_keeps_top_three() {
        let mut p = SessionPointer::new("t".into());
        let hits = (0..5)
            .map(|i| (i, "a.md".into(), format!("h{i}")))
            .collect::<Vec<_>>();
        p.advance("q", &hits, None);
        assert_eq!(p.turn, 1);
        assert_eq!(p.last_hits.len(), 3);
    }
}
