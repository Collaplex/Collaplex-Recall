use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use serde_json::json;

use collaplex_recall::aliases::{self, AddOutcome};
use collaplex_recall::chat::ChatConfig;
use collaplex_recall::config::{Config, EmbedKind};
use collaplex_recall::embed::{Embedder, HashEmbedder, HttpEmbedder};
use collaplex_recall::eval::{self, EvalOptions};
use collaplex_recall::experience::{self, AddOutcome as ExperienceOutcome};
use collaplex_recall::extract;
use collaplex_recall::feedback;
use collaplex_recall::memory::{self, AddOutcome as MemoryOutcome};
use collaplex_recall::search::{Fusion, SearchOptions, hybrid_search};
use collaplex_recall::supervisor::{self, SupervisorConfig};
use collaplex_recall::{db, ingest, log::SessionLog};

#[derive(Parser)]
#[command(
    name = "recall",
    version,
    about = "Collaplex Recall：本地优先的笔记记忆库（混合检索 + 记忆提炼 + 引用问答）"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 建库建表（幂等）
    Init,
    /// 交互式 TUI：斜杠命令 + 带引用问答
    #[command(alias = "repl")]
    Tui,
    /// 别名表：词 → 实体
    Alias {
        #[command(subcommand)]
        action: AliasCmd,
    },
    /// 记忆条：事实 + provenance
    Memory {
        #[command(subcommand)]
        action: MemoryCmd,
    },
    /// 经验：候选区 + 晋升
    Experience {
        #[command(subcommand)]
        action: ExperienceCmd,
    },
    /// 反馈记录
    Feedback {
        #[command(subcommand)]
        action: FeedbackCmd,
    },
    /// 异步提炼的 delta 队列
    Delta {
        #[command(subcommand)]
        action: DeltaCmd,
    },
    /// 记忆图边
    Edges {
        #[command(subcommand)]
        action: EdgesCmd,
    },
    /// 组件自检；--watch 变看门狗
    Supervise {
        #[arg(long)]
        watch: bool,
        #[arg(long, default_value_t = 30)]
        interval: u64,
        /// 同时拉起精排服务
        #[arg(long)]
        rerank: bool,
    },
    /// 用反馈网格搜索两段式 α
    Tune {
        #[arg(long, default_value_t = 0.5)]
        max: f64,
        #[arg(long, default_value_t = 0.05)]
        step: f64,
        /// 把最优 α 写入 settings，供 search/eval/TUI 默认读取
        #[arg(long)]
        apply: bool,
        /// 样本不足或无区分度时仍然写入
        #[arg(long)]
        force: bool,
    },
    /// 把目录下的 Markdown 笔记导入索引
    Ingest {
        /// 笔记目录（递归找 .md）
        dir: PathBuf,
        /// 记录 provenance 用的 git 仓库，默认就是 dir
        #[arg(long, value_name = "GIT_DIR")]
        repo: Option<PathBuf>,
        /// 单块最大字符数
        #[arg(long, default_value_t = 1200)]
        max_chars: usize,
    },
    /// 双路召回（向量 + 字面，RRF 融合）
    Search {
        query: String,
        /// 返回条数
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        /// 每路候选数
        #[arg(long, default_value_t = 50)]
        candidates: usize,
        /// 关闭向量召回
        #[arg(long)]
        no_vector: bool,
        /// 关闭字面召回
        #[arg(long)]
        no_lexical: bool,
        /// 向量路的 RRF 权重
        #[arg(long, default_value_t = 1.0)]
        vector_weight: f64,
        /// 字面路的 RRF 权重
        #[arg(long, default_value_t = 0.3)]
        lexical_weight: f64,
        /// 融合方式：rerank（向量召回后 BM25 重排）或 rrf（并行加权融合）
        #[arg(long, default_value = "rerank")]
        fusion: String,
        /// rerank 模式下 BM25 的权重（不填则读 settings.alpha，默认 0.05）
        #[arg(long)]
        alpha: Option<f64>,
        /// 输出 JSON（就是写进日志的结构）
        #[arg(long)]
        json: bool,
        /// 显示 chunk_id 和两路名次
        #[arg(long)]
        verbose: bool,
        /// 关闭别名展开（用于对比）
        #[arg(long)]
        no_alias: bool,
    },
    /// 批量导入 JSONL 语料（每行 {"id","text"}），用于公开评测集
    IngestJsonl {
        file: PathBuf,
        /// 笔记路径前缀，eval 用同一个值把块映射回 pid
        #[arg(long, default_value = "bench")]
        prefix: String,
        /// 单块最大字符数
        #[arg(long, default_value_t = 1200)]
        max_chars: usize,
    },
    /// 跑检索评测：queries.jsonl + qrels.jsonl
    Eval {
        queries: PathBuf,
        qrels: PathBuf,
        #[arg(long, default_value = "bench")]
        prefix: String,
        /// 截断位置
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        /// 每路候选数
        #[arg(long, default_value_t = 100)]
        candidates: usize,
        /// 只用字面召回
        #[arg(long)]
        no_vector: bool,
        /// 只用向量召回
        #[arg(long)]
        no_lexical: bool,
        #[arg(long, default_value_t = 1.0)]
        vector_weight: f64,
        #[arg(long, default_value_t = 0.3)]
        lexical_weight: f64,
        /// 融合方式：rerank 或 rrf
        #[arg(long, default_value = "rerank")]
        fusion: String,
        /// rerank 模式下 BM25 的权重（不填则读 settings.alpha，默认 0.05）
        #[arg(long)]
        alpha: Option<f64>,
        /// 用交叉编码器精排
        #[arg(long)]
        use_rerank: bool,
        /// 输出 JSON
        #[arg(long)]
        json: bool,
    },
    /// 索引概况
    Stats,
}

#[derive(Subcommand)]
enum DeltaCmd {
    /// 最近的 delta
    List {
        #[arg(long, default_value_t = 20)]
        limit: i64,
    },
    /// 合并遗留的 pending delta
    Merge {
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
}

#[derive(Subcommand)]
enum EdgesCmd {
    /// 列出图边
    List,
    /// 清理衰减后过低且长期未更新的边
    Gc {
        #[arg(long, default_value_t = 0.05)]
        min: f64,
        #[arg(long, default_value_t = 7)]
        days: i64,
    },
    /// 有限差分：边权 ±delta 内是否会跨过一跳阈值
    Sensitivity {
        #[arg(long, default_value_t = 0.1)]
        delta: f64,
    },
}

#[derive(Subcommand)]
enum FeedbackCmd {
    /// 最近反馈
    List {
        #[arg(long, default_value_t = 20)]
        limit: i64,
    },
}

#[derive(Subcommand)]
enum ExperienceCmd {
    /// 手动加一条候选经验
    Add { text: String },
    /// 列出经验
    List,
    /// 晋升候选
    Promote { id: i64 },
    /// 拒绝候选
    Reject { id: i64 },
    /// 模拟一次复用（给候选记一次相关）
    Touch { query: String },
}

#[derive(Subcommand)]
enum MemoryCmd {
    /// 新增记忆：绑定来源笔记
    Add {
        note: String,
        text: String,
        /// 绑定到具体 chunk（不填则只做笔记级 provenance）
        #[arg(long)]
        chunk: Option<i64>,
        /// 固定，源笔记变化也不自动降级
        #[arg(long)]
        pin: bool,
    },
    /// 列出记忆（含 stale）
    List,
    /// 软删除记忆
    Rm { id: i64 },
    /// 只检查 provenance，把失效记忆标为 stale
    Check,
    /// 尝试把 stale 记忆重新绑定到当前 chunk
    Reextract {
        #[arg(long, default_value_t = 100)]
        limit: usize,
        /// 用聊天模型判断旧记忆是否仍成立并重写
        #[arg(long)]
        model: bool,
    },
}

#[derive(Subcommand)]
enum AliasCmd {
    /// 新增别名：alias → entity
    Add {
        alias: String,
        entity: String,
        /// 已有别名指向别的实体时强制覆盖
        #[arg(long)]
        force: bool,
    },
    /// 列出别名
    List {
        #[arg(long)]
        filter: Option<String>,
    },
    /// 删除别名
    Rm { alias: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::from_env();

    match cli.cmd {
        Cmd::Init => {
            let client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            println!(
                "表结构就绪：notes / chunks / chunk_terms / embeddings（pgvector + 内置 BM25）"
            );
        }

        Cmd::Alias { action } => {
            let client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            match action {
                AliasCmd::Add {
                    alias,
                    entity,
                    force,
                } => match aliases::add(&client, &alias, &entity, force).await? {
                    AddOutcome::Created => println!("已新增：{alias} → {entity}"),
                    AddOutcome::Updated => println!("已更新：{alias} → {entity}"),
                    AddOutcome::Unchanged => println!("未变化：{alias} → {entity}"),
                },
                AliasCmd::List { filter } => {
                    let rows = aliases::list(&client, filter.as_deref()).await?;
                    if rows.is_empty() {
                        println!("别名表为空。");
                    }
                    for r in rows {
                        println!("{} → {}（{}）", r.alias, r.entity, r.source);
                    }
                }
                AliasCmd::Rm { alias } => {
                    if aliases::remove(&client, &alias).await? {
                        println!("已删除：{alias}");
                    } else {
                        println!("没有这个别名：{alias}");
                    }
                }
            }
        }

        Cmd::Memory { action } => {
            let client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            match action {
                MemoryCmd::Add {
                    note,
                    text,
                    chunk,
                    pin,
                } => match memory::add(&client, &text, &note, chunk, pin).await? {
                    MemoryOutcome::Created(id) => {
                        println!("已新增记忆 #{id}（来源 {note}）")
                    }
                    MemoryOutcome::Existing(id) => {
                        println!("已存在记忆 #{id}（{note}）")
                    }
                },
                MemoryCmd::List => {
                    let rows = memory::list(&client).await?;
                    if rows.is_empty() {
                        println!("没有记忆。");
                    }
                    for r in rows {
                        let stale = if r.status == "stale" { " [stale]" } else { "" };
                        let pin = if r.pinned { " [pinned]" } else { "" };
                        println!(
                            "#{} {}{}{}：{}（来源 {}）",
                            r.id,
                            r.status,
                            stale,
                            pin,
                            r.text,
                            r.note_path.unwrap_or_default()
                        );
                    }
                }
                MemoryCmd::Rm { id } => {
                    if memory::remove(&client, id).await? {
                        println!("已软删除 #{id}");
                    } else {
                        println!("没有这个记忆：#{id}");
                    }
                }
                MemoryCmd::Check => {
                    let n = memory::refresh(&client).await?;
                    println!("检查完成，降级 {n} 条");
                }
                MemoryCmd::Reextract { limit, model } => {
                    if model {
                        let chat = ChatConfig::from_env();
                        let report = collaplex_recall::reextract::reextract_with_model(
                            &client, &chat, limit,
                        )
                        .await?;
                        for e in &report.errors {
                            eprintln!("警告: {e}");
                        }
                        println!(
                            "模型重提炼：重写 {} 条，放弃 {} 条",
                            report.rewritten, report.dropped
                        );
                    } else {
                        let (revived, left) = memory::reextract(&client, limit).await?;
                        println!("重提炼：恢复 {revived} 条，仍失效 {left} 条");
                    }
                }
            }
        }

        Cmd::Experience { action } => {
            let client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            match action {
                ExperienceCmd::Add { text } => {
                    match experience::add_candidate(&client, &text, "manual", 0).await? {
                        ExperienceOutcome::Created(id) => println!("已新增经验候选 #{id}"),
                        ExperienceOutcome::Existing(id) => println!("已存在经验 #{id}"),
                    }
                }
                ExperienceCmd::List => {
                    let rows = experience::list(&client).await?;
                    if rows.is_empty() {
                        println!("没有经验。");
                    }
                    for r in rows {
                        println!("#{} {} uses={}：{}", r.id, r.status, r.uses, r.text);
                    }
                }
                ExperienceCmd::Promote { id } => {
                    if experience::promote(&client, id).await? {
                        println!("已晋升 #{id}");
                    } else {
                        println!("没有可晋升的候选：#{id}");
                    }
                }
                ExperienceCmd::Reject { id } => {
                    if experience::reject(&client, id).await? {
                        println!("已拒绝 #{id}");
                    } else {
                        println!("没有这条经验：#{id}");
                    }
                }
                ExperienceCmd::Touch { query } => {
                    let ids: Vec<i64> = experience::match_candidates(&client, &query)
                        .await?
                        .into_iter()
                        .map(|(id, _)| id)
                        .collect();
                    let promoted =
                        experience::reward(&client, &ids, experience::promote_after()).await?;
                    if promoted.is_empty() {
                        println!("匹配 {} 条候选，本次无晋升。", ids.len());
                    } else {
                        for p in promoted {
                            println!("已晋升：{p}");
                        }
                    }
                }
            }
        }

        Cmd::Feedback { action } => {
            let client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            match action {
                FeedbackCmd::List { limit } => {
                    for r in feedback::list(&client, limit).await? {
                        println!(
                            "#{} {} turn={} {} {}",
                            r.id,
                            r.created_at,
                            r.turn.unwrap_or_default(),
                            r.kind,
                            r.target.unwrap_or_default()
                        );
                    }
                }
            }
        }

        Cmd::Delta { action } => {
            let client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            match action {
                DeltaCmd::List { limit } => {
                    let rows = client
                        .query(
                            "select id, session_id, turn, status, created_at::text,
                                    payload->>'result'
                             from memory_deltas
                             order by id desc
                             limit $1",
                            &[&limit],
                        )
                        .await?;
                    for r in rows {
                        println!(
                            "#{} {} turn={} {} {}",
                            r.get::<_, i64>(0),
                            r.get::<_, String>(4),
                            r.get::<_, Option<i32>>(2).unwrap_or(0),
                            r.get::<_, String>(3),
                            r.get::<_, Option<String>>(5).unwrap_or_default()
                        );
                    }
                }
                DeltaCmd::Merge { limit } => {
                    let ec = extract::ExtractConfig::from_env();
                    let reports =
                        extract::merge_pending(&client, limit, ec.max_memories, ec.max_aliases)
                            .await?;
                    for r in &reports {
                        println!(
                            "delta #{}：记忆 +{} 别名 +{} 经验 +{}",
                            r.delta_id, r.memories_added, r.aliases_added, r.experiences_added
                        );
                    }
                    println!("合并 {} 条 pending", reports.len());
                }
            }
        }

        Cmd::Edges { action } => {
            let client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            match action {
                EdgesCmd::List => {
                    let rows = client
                        .query(
                            "select a_id, b_id, round(weight::numeric, 3)::text, t_updated::text
                             from memory_edges
                             order by a_id, b_id",
                            &[],
                        )
                        .await?;
                    for r in rows {
                        println!(
                            "{} - {}  w={}  {}",
                            r.get::<_, i64>(0),
                            r.get::<_, i64>(1),
                            r.get::<_, String>(2),
                            r.get::<_, String>(3)
                        );
                    }
                }
                EdgesCmd::Gc { min, days } => {
                    let n = memory::gc_edges(&client, min, days).await?;
                    println!("清理 {n} 条边（衰减后 < {min} 且 {days} 天未更新）");
                }
                EdgesCmd::Sensitivity { delta } => {
                    let rows = memory::edge_sensitivity(&client, delta).await?;
                    let flips = rows.iter().filter(|e| e.flips).count();
                    println!(
                        "边 {} 条，±{delta} 内会翻转 {} 条（阈值 {:.3}）",
                        rows.len(),
                        flips,
                        memory::edge_min()
                    );
                    for e in rows.iter().take(20) {
                        println!(
                            "{} - {}  w={:.3} decayed={:.3} margin={:.3}{}",
                            e.a_id,
                            e.b_id,
                            e.weight,
                            e.decayed,
                            e.margin,
                            if e.flips { "  [会翻转]" } else { "" }
                        );
                    }
                }
            }
        }

        Cmd::Supervise {
            watch,
            interval,
            rerank,
        } => {
            let want_embed = matches!(cfg.embed, EmbedKind::Http { .. })
                && supervisor::env_url_is_local("NOTE_EMBED_URL", "http://127.0.0.1:8081/v1");
            let want_chat =
                supervisor::env_url_is_local("NOTE_CHAT_URL", "http://127.0.0.1:8082/v1");
            let want_rerank = rerank
                || std::env::var("NOTE_RERANK")
                    .map(|v| matches!(v.to_ascii_lowercase().as_str(), "on" | "1" | "true"))
                    .unwrap_or(false);
            let sc = SupervisorConfig::new(
                cfg.log_dir.clone(),
                cfg.database_url.clone(),
                want_embed,
                want_chat,
                want_rerank,
            );
            if watch {
                supervisor::watch(&sc, interval).await;
            } else if !supervisor::ensure_all(&sc, true).await {
                bail!(
                    "有组件未就绪，见 {}",
                    cfg.log_dir.join("supervisor.log").display()
                );
            }
        }

        Cmd::Tune {
            max,
            step,
            apply,
            force,
        } => {
            let client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            let report = collaplex_recall::tune::tune(&client, &cfg.log_dir, max, step).await?;
            println!(
                "反馈轮次 {}｜正样本 {}｜负样本 {}｜样本告警 {}",
                report.turns_with_feedback,
                report.positive_events,
                report.negative_events,
                report.warnings.len()
            );
            for w in &report.warnings {
                eprintln!("警告: {w}");
            }
            println!(
                "最优 α={:.2}（正样本 MRR {:.3} - 负样本 MRR {:.3} = {:.3}）",
                report.best_alpha,
                report
                    .grid
                    .iter()
                    .find(|(a, _, _, _)| (*a - report.best_alpha).abs() < 1e-9)
                    .map(|(_, p, _, _)| *p)
                    .unwrap_or(0.0),
                report
                    .grid
                    .iter()
                    .find(|(a, _, _, _)| (*a - report.best_alpha).abs() < 1e-9)
                    .map(|(_, _, n, _)| *n)
                    .unwrap_or(0.0),
                report.best_score
            );
            let unreliable = report
                .warnings
                .iter()
                .any(|w| w.contains("无法区分") || w.contains("少于 5"));
            if apply {
                if unreliable && !force {
                    eprintln!("样本不足或无区分度，拒绝写入；要覆盖加 --force。");
                } else {
                    collaplex_recall::settings::set(
                        &client,
                        "alpha",
                        &format!("{:.4}", report.best_alpha),
                    )
                    .await?;
                    println!("已写入 settings.alpha={:.4}", report.best_alpha);
                }
            }
        }

        Cmd::Tui => {
            let client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            drop(client);
            let embedder: Arc<dyn Embedder> = Arc::from(build_embedder(&cfg)?);
            let chat = ChatConfig::from_env();
            collaplex_recall::tui::run(cfg, embedder, chat).await?;
        }

        Cmd::Ingest {
            dir,
            repo,
            max_chars,
        } => {
            let mut client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            let embedder = build_embedder(&cfg)?;
            let repo = repo.as_deref().or(Some(dir.as_path()));

            let stats =
                ingest::ingest_dir(&mut client, embedder.as_ref(), &dir, repo, max_chars).await?;

            for w in &stats.warnings {
                eprintln!("警告: {w}");
            }
            println!(
                "扫描 {} 个文件：写入 {} 篇（{} 块），未变 {} 篇，警告 {} 条",
                stats.files_seen,
                stats.notes_written,
                stats.chunks_written,
                stats.notes_skipped,
                stats.warnings.len()
            );
            println!("嵌入模型: {}", embedder.model_id());
        }

        Cmd::IngestJsonl {
            file,
            prefix,
            max_chars,
        } => {
            let mut client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            let embedder = build_embedder(&cfg)?;

            println!(
                "导入 {}（前缀 {}），嵌入模型 {}",
                file.display(),
                prefix,
                embedder.model_id()
            );
            let stats =
                ingest::ingest_jsonl(&mut client, embedder.as_ref(), &file, &prefix, max_chars)
                    .await?;

            for w in stats.warnings.iter().take(20) {
                eprintln!("警告: {w}");
            }
            println!(
                "语料 {} 条：写入 {} 篇（{} 块），未变 {} 篇，警告 {} 条",
                stats.files_seen,
                stats.notes_written,
                stats.chunks_written,
                stats.notes_skipped,
                stats.warnings.len()
            );
        }

        Cmd::Eval {
            queries,
            qrels,
            prefix,
            k,
            candidates,
            no_vector,
            no_lexical,
            vector_weight,
            lexical_weight,
            fusion,
            alpha,
            use_rerank,
            json,
        } => {
            if no_vector && no_lexical {
                bail!("两路都关了，没有可检索的东西");
            }
            let client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            let embedder = if no_vector {
                None
            } else {
                Some(build_embedder(&cfg)?)
            };
            let alpha = match alpha {
                Some(a) => a,
                None => collaplex_recall::settings::get(&client, "alpha")
                    .await?
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.05),
            };
            let opts = EvalOptions {
                k,
                candidates,
                use_vector: !no_vector,
                use_lexical: !no_lexical,
                fusion: build_fusion(&fusion, vector_weight, lexical_weight, alpha)?,
                use_rerank,
                prefix,
            };

            let report =
                eval::run_eval(&client, embedder.as_deref(), &queries, &qrels, &opts).await?;

            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
                return Ok(());
            }

            let mode = match (opts.use_vector, opts.use_lexical) {
                (true, true) => "hybrid",
                (true, false) => "vector",
                (false, true) => "lexical",
                _ => unreachable!(),
            };
            let fusion = match opts.fusion {
                Fusion::Rrf {
                    vector_weight,
                    lexical_weight,
                } => format!("rrf(v={vector_weight}, l={lexical_weight})"),
                Fusion::Rerank { alpha } => format!("rerank(alpha={alpha})"),
            };
            println!(
                "模式 {mode}｜融合 {fusion}｜有标注查询 {}/{}｜k={}",
                report.queries_scored, report.queries_total, report.k
            );
            println!(
                "recall@{} {:.4}｜hit@{} {:.4}｜MRR {:.4}｜nDCG {} {:.4}",
                report.k,
                report.recall_at_k,
                report.k,
                report.hit_at_k,
                report.mrr_at_k,
                report.k,
                report.ndcg_at_k
            );
            println!(
                "无相关命中 {}｜完全无结果 {}",
                report.queries_without_relevant_hit, report.queries_without_any_hit
            );
            for w in &report.warnings {
                eprintln!("警告: {w}");
            }
        }

        Cmd::Search {
            query,
            k,
            candidates,
            no_vector,
            no_lexical,
            vector_weight,
            lexical_weight,
            fusion,
            alpha,
            json,
            verbose,
            no_alias,
        } => {
            if no_vector && no_lexical {
                bail!("两路都关了，没有可检索的东西");
            }
            let client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;

            let embedder = if no_vector {
                None
            } else {
                Some(build_embedder(&cfg)?)
            };
            let alpha = match alpha {
                Some(a) => a,
                None => collaplex_recall::settings::get(&client, "alpha")
                    .await?
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.05),
            };
            let opts = SearchOptions {
                k,
                candidates,
                use_vector: !no_vector,
                use_lexical: !no_lexical,
                fusion: build_fusion(&fusion, vector_weight, lexical_weight, alpha)?,
            };

            let (query_used, alias_hits) = if no_alias {
                (query.clone(), Vec::new())
            } else {
                match aliases::expand(&client, &query).await {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("警告: 别名展开失败: {e}");
                        (query.clone(), Vec::new())
                    }
                }
            };
            if !alias_hits.is_empty() && !json {
                println!(
                    "别名：{}",
                    alias_hits
                        .iter()
                        .map(|h| format!("{} → {}", h.alias, h.entity))
                        .collect::<Vec<_>>()
                        .join("，")
                );
            }

            let outcome = hybrid_search(&client, embedder.as_deref(), &query_used, &opts).await?;

            let log = SessionLog::new(&cfg.log_dir)?;
            log.append(&json!({
                "ts": chrono::Local::now().to_rfc3339(),
                "kind": "search",
                "query": query,
                "query_used": outcome.query,
                "aliases": alias_hits,
                "embedder": outcome.embedder,
                "k": outcome.k,
                "candidates": { "vector": outcome.candidates.vector, "lexical": outcome.candidates.lexical },
                "elapsed_ms": outcome.elapsed_ms,
                "warnings": outcome.warnings,
                "hits": outcome.hits.iter().map(|h| json!({
                    "rank": h.rank,
                    "chunk_id": h.chunk_id,
                    "path": h.path,
                    "heading": h.heading,
                    "rrf": h.rrf,
                    "vector_rank": h.vector_rank,
                    "vector_score": h.vector_score,
                    "lexical_rank": h.lexical_rank,
                    "lexical_score": h.lexical_score,
                })).collect::<Vec<_>>(),
            }))?;

            if json {
                println!("{}", serde_json::to_string_pretty(&outcome)?);
                return Ok(());
            }

            for w in &outcome.warnings {
                eprintln!("警告: {w}");
            }
            println!(
                "候选 向量 {}/ 字面 {}｜嵌入 {}｜{} ms｜日志 {}",
                outcome.candidates.vector,
                outcome.candidates.lexical,
                outcome.embedder,
                outcome.elapsed_ms,
                log.path().display()
            );
            if outcome.hits.is_empty() {
                println!("没有命中。");
            }
            for h in &outcome.hits {
                let heading = if h.heading.is_empty() {
                    "（无标题）".to_string()
                } else {
                    h.heading.clone()
                };
                let score_label = match opts.fusion {
                    Fusion::Rerank { .. } => "score",
                    Fusion::Rrf { .. } => "rrf",
                };
                let mut meta = format!("{score_label}={:.4}", h.rrf);
                if let Some(r) = h.vector_rank {
                    let s = h.vector_score.unwrap_or_default();
                    meta.push_str(&format!("  vec=#{r} {s:.3}"));
                }
                if let Some(r) = h.lexical_rank {
                    let s = h.lexical_score.unwrap_or_default();
                    meta.push_str(&format!("  lex=#{r} {s:.3}"));
                }
                if verbose {
                    meta.push_str(&format!("  chunk={}", h.chunk_id));
                }
                println!("[{}] {} › {}", h.rank, h.path, heading);
                println!("    {meta}");
                println!("    {}", snippet(&h.text, 160));
            }
        }

        Cmd::Stats => {
            let client = db::connect(&cfg.database_url).await?;
            db::init_schema(&client).await?;
            let row = client
                .query_one(
                    "select (select count(*) from notes),
                            (select count(*) from chunks),
                            (select count(*) from embeddings)",
                    &[],
                )
                .await?;
            println!(
                "笔记 {} 篇｜块 {}｜向量 {}",
                row.get::<_, i64>(0),
                row.get::<_, i64>(1),
                row.get::<_, i64>(2)
            );
            let rows = client
                .query(
                    "select model, dim, count(*) from embeddings group by model, dim order by model",
                    &[],
                )
                .await?;
            for r in rows {
                println!(
                    "  模型 {}（{} 维）：{} 条",
                    r.get::<_, String>(0),
                    r.get::<_, i32>(1),
                    r.get::<_, i64>(2)
                );
            }
        }
    }

    Ok(())
}

fn build_fusion(name: &str, vector_weight: f64, lexical_weight: f64, alpha: f64) -> Result<Fusion> {
    match name {
        "rrf" => Ok(Fusion::Rrf {
            vector_weight,
            lexical_weight,
        }),
        "rerank" => Ok(Fusion::Rerank { alpha }),
        other => bail!("未知 fusion: {other}（可选 rrf / rerank）"),
    }
}

fn build_embedder(cfg: &Config) -> Result<Box<dyn Embedder>> {
    match &cfg.embed {
        EmbedKind::Hash { dim } => Ok(Box::new(HashEmbedder::new(*dim))),
        EmbedKind::Http {
            base_url,
            model,
            dim,
            api_key,
            query_prefix,
            doc_prefix,
        } => Ok(Box::new(HttpEmbedder::new(
            base_url.clone(),
            model.clone(),
            *dim,
            api_key.clone(),
            query_prefix.clone(),
            doc_prefix.clone(),
            60,
        )?)),
    }
}

fn snippet(text: &str, max_chars: usize) -> String {
    let one_line: String = text
        .chars()
        .map(|c| if c == '\n' { ' ' } else { c })
        .collect();
    let mut s: String = one_line.chars().take(max_chars).collect();
    if one_line.chars().count() > max_chars {
        s.push('…');
    }
    s
}
