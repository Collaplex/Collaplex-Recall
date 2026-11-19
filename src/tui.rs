use std::io;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use ratatui::Frame;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Clear, Paragraph, Wrap};
use serde_json::json;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::task::JoinHandle;
use unicode_width::UnicodeWidthStr;

use crate::aliases::{self, AddOutcome, AliasHit};
use crate::chat::{ChatConfig, ChatProgress, ChatResult, stream_chat};
use crate::config::Config;
use crate::db;
use crate::embed::Embedder;
use crate::experience;
use crate::extract::{self, ExtractConfig, ExtractMode, Source};
use crate::feedback;
use crate::inbox;
use crate::log::SessionLog;
use crate::memory::{self, MemoryHit};
use crate::propose;
use crate::rerank::{self, RerankConfig};
use crate::rewrite::{self, RewriteConfig};
use crate::search::{Fusion, Hit, SearchOptions, SearchOutcome, hybrid_search};
use crate::select::{self, SelectConfig};
use crate::session::{ResolvedRef, SessionPointer};
use crate::settings;
use crate::token::estimate_tokens;
use crate::verify::{self, VerifyConfig};

const SYSTEM_PROMPT: &str = "你是一个本地笔记助手。只能依据用户提供的笔记片段回答，\
引用格式为 [n]，每个结论都要标注来源编号。片段不足时直接说明，不要编造。用中文回答。";

const COMMANDS: &[(&str, &str)] = &[
    ("/search", "只检索，看两路名次和分数"),
    ("/stats", "索引概况"),
    ("/ingest", "导入 Markdown 目录"),
    ("/note", "追加到收件箱"),
    ("/inbox", "查看收件箱"),
    ("/promote", "收件箱条目提升进正式笔记"),
    ("/inbox-drop", "丢弃收件箱条目"),
    ("/alias", "别名：add / list / del"),
    ("/memory", "记忆：add / list / rm / check / reextract"),
    ("/experience", "经验：add / list / promote / reject"),
    ("/adopt", "采纳上一轮答案"),
    ("/reject", "纠正上一轮答案"),
    ("/cite", "打开并记录引用点击"),
    ("/feedback", "最近反馈"),
    ("/propose", "生成补丁 diff 提案"),
    ("/proposals", "待处理提案"),
    ("/apply", "写入提案（唯一写笔记路径）"),
    ("/discard", "拒绝提案"),
    ("/last", "重看上一轮检索"),
    ("/why", "解释上一轮检索"),
    ("/clear", "清屏"),
    ("/help", "命令帮助"),
    ("/quit", "退出并关闭本机模型组件"),
];

pub async fn run(cfg: Config, embedder: Arc<dyn Embedder>, chat: ChatConfig) -> Result<()> {
    let want_embed = matches!(cfg.embed, crate::config::EmbedKind::Http { .. })
        && crate::supervisor::env_url_is_local("NOTE_EMBED_URL", "http://127.0.0.1:8081/v1");
    let want_chat =
        crate::supervisor::env_url_is_local("NOTE_CHAT_URL", "http://127.0.0.1:8082/v1");
    let want_rerank = std::env::var("NOTE_RERANK")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "on" | "1" | "true"))
        .unwrap_or(false);
    let sc = crate::supervisor::SupervisorConfig::new(
        cfg.log_dir.clone(),
        cfg.database_url.clone(),
        want_embed,
        want_chat,
        want_rerank,
    );

    // 启动自检：本机组件没跑就拉起（NOTE_SELFCHECK=off 可关）。
    let selfcheck_off = std::env::var("NOTE_SELFCHECK")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "off" | "0" | "false"))
        .unwrap_or(false);
    if !selfcheck_off && !crate::supervisor::ensure_all(&sc, true).await {
        eprintln!("组件自检未全部通过，继续启动 TUI");
    }

    // 启动时检查 git HEAD；变了就增量 ingest 一次。
    if let Err(e) = sync_notes(&cfg, embedder.as_ref()).await {
        eprintln!("启动同步失败: {e:#}");
    }
    enable_raw_mode().context("开启 raw 模式失败")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = event_loop(&mut terminal, cfg, embedder, chat).await;
    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();

    let quit = matches!(&result, Ok(true));
    if quit && quit_shutdown_on() {
        println!("正在按顺序关闭本机模型组件…");
        crate::supervisor::shutdown_all(&sc).await;
    }
    result.map(|_| ())
}

fn quit_shutdown_on() -> bool {
    std::env::var("NOTE_QUIT_SHUTDOWN")
        .map(|v| !matches!(v.to_ascii_lowercase().as_str(), "off" | "0" | "false"))
        .unwrap_or(true)
}

async fn sync_notes(cfg: &Config, embedder: &dyn Embedder) -> Result<()> {
    let mut client = db::connect(&cfg.database_url).await?;
    db::init_schema(&client).await?;
    let Some(root) = settings::get(&client, "notes_root").await? else {
        return Ok(());
    };
    let root = std::path::PathBuf::from(root);
    let Some(head) = crate::ingest::git_head(&root) else {
        return Ok(());
    };
    let last = settings::get(&client, "last_commit").await?;
    if last.as_deref() == Some(head.as_str()) {
        return Ok(());
    }
    let stats = crate::ingest::ingest_dir(&mut client, embedder, &root, Some(&root), 1200).await?;
    eprintln!(
        "启动同步 {}：写入 {} 篇 / 新增 {} 块",
        &head[..head.len().min(8)],
        stats.notes_written,
        stats.chunks_written
    );
    Ok(())
}

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    cfg: Config,
    embedder: Arc<dyn Embedder>,
    chat: ChatConfig,
) -> Result<bool> {
    let (tx, mut rx) = unbounded_channel::<Msg>();
    let (ev_tx, mut ev_rx) = unbounded_channel::<Event>();
    // 用普通线程而不是 spawn_blocking：后者会阻塞 tokio 运行时退出，
    // 因为 event::read() 永远等不到下一次按键。
    std::thread::spawn(move || {
        while let Ok(ev) = event::read() {
            if ev_tx.send(ev).is_err() {
                break;
            }
        }
    });

    let mut app = App::new(cfg, embedder, chat, tx);
    let mut tick = tokio::time::interval(Duration::from_millis(100));

    loop {
        terminal.draw(|f| app.draw(f))?;
        tokio::select! {
            Some(ev) = ev_rx.recv() => app.on_event(ev),
            Some(msg) = rx.recv() => app.on_msg(msg),
            _ = tick.tick() => {}
        }
        if app.quit {
            break;
        }
    }

    if let Some(task) = app.task.take() {
        task.abort();
    }
    Ok(app.quit)
}

enum Msg {
    Lines(Vec<Line<'static>>),
    SearchDone {
        query: String,
        outcome: Box<SearchOutcome>,
    },
    ChatDone {
        lines: Vec<Line<'static>>,
        pointer: SessionPointer,
        feedback: Option<FeedbackCtx>,
    },
    Status(ChatProgress),
    Phase(String),
    Clarify(ClarifyRequest),
    AliasLearned {
        question: String,
        term: String,
        entity: String,
    },
    ClarifyFailed {
        request: ClarifyRequest,
        error: String,
    },
    QueueExtract(PendingExtract),
    FlushExtract,
    Busy(bool),
    Error(String),
}

#[derive(Clone)]
struct PendingExtract {
    question: String,
    answer: String,
    sources: Vec<Source>,
    session_id: String,
    turn: u32,
}

#[derive(Clone)]
struct ClarifyRequest {
    question: String,
    term: String,
}

#[derive(Clone)]
struct FeedbackCtx {
    session_id: String,
    turn: u32,
    question: String,
    sources: Vec<Source>,
    memory_ids: Vec<i64>,
    experience_candidates: Vec<i64>,
}

#[derive(Clone, Default)]
struct StatusView {
    tokens: u64,
    prompt_tokens: u32,
    kv_used: u32,
    kv_ctx: u32,
    elapsed_ms: u128,
    context_used: u32,
    context_budget: u32,
    finished: bool,
}

struct App {
    cfg: Config,
    embedder: Arc<dyn Embedder>,
    chat: ChatConfig,
    rewrite: RewriteConfig,
    select: SelectConfig,
    extract: ExtractConfig,
    extract_queue: Vec<PendingExtract>,
    idle_timer: Option<JoinHandle<()>>,
    verify: VerifyConfig,
    rerank: RerankConfig,
    tx: UnboundedSender<Msg>,
    input: Vec<char>,
    cursor: usize,
    menu_index: usize,
    lines: Vec<Line<'static>>,
    scroll_back: usize,
    busy: bool,
    busy_label: String,
    status: Option<StatusView>,
    last_status: Option<StatusView>,
    task: Option<JoinHandle<()>>,
    last_search: Option<(String, SearchOutcome)>,
    session: SessionPointer,
    last_feedback: Option<FeedbackCtx>,
    pending_clarify: Option<ClarifyRequest>,
    log: Option<SessionLog>,
    quit: bool,
}

impl App {
    fn new(
        cfg: Config,
        embedder: Arc<dyn Embedder>,
        chat: ChatConfig,
        tx: UnboundedSender<Msg>,
    ) -> Self {
        let log = SessionLog::new(&cfg.log_dir).ok();
        let session_id = format!(
            "{}-{}",
            chrono::Local::now().format("%Y%m%d-%H%M%S"),
            std::process::id()
        );
        let mut app = Self {
            session: SessionPointer::new(session_id),
            cfg,
            embedder,
            chat,
            rewrite: RewriteConfig::from_env(),
            select: SelectConfig::from_env(),
            extract: ExtractConfig::from_env(),
            extract_queue: Vec::new(),
            idle_timer: None,
            verify: VerifyConfig::from_env(),
            rerank: RerankConfig::from_env(),
            tx,
            input: Vec::new(),
            cursor: 0,
            menu_index: 0,
            lines: Vec::new(),
            scroll_back: 0,
            busy: false,
            busy_label: String::new(),
            status: None,
            last_status: None,
            task: None,
            last_search: None,
            last_feedback: None,
            pending_clarify: None,
            log,
            quit: false,
        };
        let width = ratatui::crossterm::terminal::size()
            .map(|(w, _)| w as usize)
            .unwrap_or(80);
        let embed = std::env::var("NOTE_EMBED_MODEL").unwrap_or_else(|_| "hash".into());
        let chat_model = std::env::var("NOTE_CHAT_MODEL").unwrap_or_else(|_| "local-chat".into());
        app.lines.push(Line::from(Span::styled(
            format!(
                "collaplex-recall {} ({}) [Rust + PostgreSQL 18/pgvector] on {}",
                env!("CARGO_PKG_VERSION"),
                chrono::Local::now().format("%Y-%m-%d"),
                std::env::consts::OS
            ),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )));
        app.push_dim(format!("embedding: {embed} | chat: {chat_model}"));
        app.push_dim("Type \"/\" to browse commands, \"/help\" for help, \"/quit\" to exit.");
        app.lines.push(Line::from(Span::styled(
            "=".repeat(width.max(1)),
            Style::default().fg(Color::DarkGray),
        )));
        app.lines.push(Line::from(""));
        app
    }

    fn on_event(&mut self, ev: Event) {
        let Event::Key(key) = ev else {
            return;
        };
        if key.kind != KeyEventKind::Press {
            return;
        }
        match (key.modifiers, key.code) {
            (KeyModifiers::CONTROL, KeyCode::Char('c')) => {
                if self.busy {
                    self.abort();
                    self.push_dim("已中断");
                } else {
                    self.quit = true;
                }
            }
            (KeyModifiers::CONTROL, KeyCode::Char('l')) => self.lines.clear(),
            (_, KeyCode::Enter) => self.submit(),
            (_, KeyCode::Backspace) => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.input.remove(self.cursor);
                    self.menu_index = 0;
                }
            }
            (_, KeyCode::Delete) => {
                if self.cursor < self.input.len() {
                    self.input.remove(self.cursor);
                    self.menu_index = 0;
                }
            }
            (_, KeyCode::Left) => self.cursor = self.cursor.saturating_sub(1),
            (_, KeyCode::Right) => {
                if self.cursor < self.input.len() {
                    self.cursor += 1;
                }
            }
            (_, KeyCode::Up) => {
                let n = self.menu_matches().len();
                if n > 0 {
                    self.menu_index = if self.menu_index == 0 {
                        n - 1
                    } else {
                        self.menu_index - 1
                    };
                } else {
                    self.scroll_back = self.scroll_back.saturating_add(1);
                }
            }
            (_, KeyCode::Down) => {
                let n = self.menu_matches().len();
                if n > 0 {
                    self.menu_index = (self.menu_index + 1) % n;
                } else {
                    self.scroll_back = self.scroll_back.saturating_sub(1);
                }
            }
            (_, KeyCode::Tab) => {
                self.complete_command();
            }
            (_, KeyCode::Home) => self.cursor = 0,
            (_, KeyCode::End) => self.cursor = self.input.len(),
            (_, KeyCode::PageUp) => self.scroll_back += 5,
            (_, KeyCode::PageDown) => self.scroll_back = self.scroll_back.saturating_sub(5),
            (_, KeyCode::Char(c)) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.input.insert(self.cursor, c);
                self.cursor += 1;
                self.menu_index = 0;
            }
            _ => {}
        }
    }

    fn on_msg(&mut self, msg: Msg) {
        match msg {
            Msg::Lines(lines) => {
                self.lines.extend(lines);
                self.scroll_back = 0;
            }
            Msg::SearchDone { query, outcome } => {
                let hit_meta: Vec<(i64, String, String)> = outcome
                    .hits
                    .iter()
                    .map(|h| (h.chunk_id, h.path.clone(), h.heading.clone()))
                    .collect();
                self.session.remember(&query, &hit_meta);
                let lines = format_search(&query, &outcome);
                self.lines.extend(lines);
                self.scroll_back = 0;
                self.last_search = Some((query, *outcome));
            }
            Msg::ChatDone {
                lines,
                pointer,
                feedback,
            } => {
                self.lines.extend(lines);
                self.scroll_back = 0;
                self.session = pointer;
                self.last_feedback = feedback;
            }
            Msg::Clarify(req) => {
                self.pending_clarify = Some(req);
                self.push_dim("直接输入实体，回车跳过");
            }
            Msg::AliasLearned {
                question,
                term,
                entity,
            } => {
                self.push_dim(format!("已学习别名：{term} → {entity}"));
                self.spawn_chat(question, false);
            }
            Msg::ClarifyFailed { request, error } => {
                self.push_red(format!("别名写入失败: {error}"));
                self.pending_clarify = Some(request);
                self.busy = false;
                self.status = None;
                self.busy_label.clear();
                self.task = None;
            }
            Msg::Phase(p) => self.busy_label = p,
            Msg::Status(p) => {
                self.status = Some(StatusView {
                    tokens: p.tokens,
                    prompt_tokens: p.prompt_tokens,
                    kv_used: p.kv_used,
                    kv_ctx: p.kv_ctx,
                    elapsed_ms: p.elapsed_ms,
                    context_used: p.context_used,
                    context_budget: p.context_budget,
                    finished: false,
                });
            }
            Msg::QueueExtract(p) => {
                self.extract_queue.push(p);
                match self.extract.mode {
                    ExtractMode::EveryN(n) => {
                        if self.extract_queue.len() >= n {
                            self.flush_extracts();
                        }
                    }
                    ExtractMode::Idle(ms) => {
                        if let Some(t) = self.idle_timer.take() {
                            t.abort();
                        }
                        let tx = self.tx.clone();
                        self.idle_timer = Some(tokio::spawn(async move {
                            tokio::time::sleep(Duration::from_millis(ms)).await;
                            let _ = tx.send(Msg::FlushExtract);
                        }));
                        if self.extract_queue.len() >= 8 {
                            self.flush_extracts();
                        }
                    }
                    ExtractMode::Immediate => self.flush_extracts(),
                }
            }
            Msg::FlushExtract => {
                self.idle_timer = None;
                self.flush_extracts();
            }
            Msg::Busy(busy) => {
                self.busy = busy;
                if !busy {
                    if let Some(mut s) = self.status.take() {
                        s.finished = true;
                        self.last_status = Some(s);
                    }
                    self.busy_label.clear();
                    self.task = None;
                }
            }
            Msg::Error(e) => {
                self.push_red(e);
            }
        }
    }

    fn menu_matches(&self) -> Vec<(&'static str, &'static str)> {
        let text: String = self.input.iter().collect();
        if !text.starts_with('/') || text.contains(char::is_whitespace) {
            return Vec::new();
        }
        let q = text.trim_start_matches('/').to_ascii_lowercase();
        COMMANDS
            .iter()
            .filter(|(name, _)| {
                name.trim_start_matches('/')
                    .to_ascii_lowercase()
                    .contains(&q)
            })
            .copied()
            .collect()
    }

    fn complete_command(&mut self) -> bool {
        let menu = self.menu_matches();
        if menu.is_empty() {
            return false;
        }
        let idx = self.menu_index.min(menu.len() - 1);
        let (name, _) = menu[idx];
        self.input = format!("{name} ").chars().collect();
        self.cursor = self.input.len();
        self.menu_index = 0;
        true
    }

    fn submit(&mut self) {
        if let Some(t) = self.idle_timer.take() {
            t.abort();
        }
        let line: String = self.input.iter().collect();
        if line.trim().is_empty() && self.pending_clarify.is_none() {
            return;
        }
        // 输入未完成的命令时，Enter 先补全，再按一次才执行。
        if !line.contains(char::is_whitespace) && line.starts_with('/') {
            let menu = self.menu_matches();
            if !menu.is_empty() {
                let idx = self.menu_index.min(menu.len() - 1);
                if line.trim() != menu[idx].0 {
                    self.complete_command();
                    return;
                }
            }
        }
        self.input.clear();
        self.cursor = 0;
        if self.busy {
            self.push_dim("当前任务还没结束（Ctrl-C 中断）");
            return;
        }
        if let Some(req) = self.pending_clarify.take() {
            let text = if line.trim().is_empty() {
                "（跳过）".to_string()
            } else {
                line.trim().to_string()
            };
            self.push_user(&text);
            self.handle_clarify(req, text);
            return;
        }
        self.push_user(&line);
        if let Some(rest) = line.strip_prefix('/') {
            self.command(rest);
        } else {
            self.spawn_chat(line, true);
        }
    }

    fn handle_clarify(&mut self, req: ClarifyRequest, answer: String) {
        if answer == "（跳过）" || answer == "/skip" {
            self.push_dim("跳过别名，按原问题继续");
            self.spawn_chat(req.question, false);
            return;
        }
        if answer.starts_with('/') {
            self.push_dim("请输入实体文本，或回车跳过");
            self.pending_clarify = Some(req);
            return;
        }

        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        let term = req.term.clone();
        let question = req.question.clone();
        let entity = answer.clone();
        let req_for_err = req;
        self.busy = true;
        self.busy_label = "学习别名…".into();
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                db::init_schema(&client).await?;
                aliases::add(&client, &term, &entity, false).await
            }
            .await;
            match res {
                Ok(_) => {
                    let _ = tx.send(Msg::AliasLearned {
                        question,
                        term,
                        entity,
                    });
                }
                Err(e) => {
                    let _ = tx.send(Msg::ClarifyFailed {
                        request: req_for_err,
                        error: format!("{e:#}"),
                    });
                }
            }
        });
        self.task = Some(handle);
    }

    fn command(&mut self, raw: &str) {
        let (cmd, arg) = match raw.split_once(' ') {
            Some((c, a)) => (c, a.trim()),
            None => (raw, ""),
        };
        match cmd {
            "help" => self.help(),
            "quit" | "q" => self.quit = true,
            "clear" => self.lines.clear(),
            "search" => {
                if arg.is_empty() {
                    self.push_dim("用法：/search 关键词");
                } else {
                    self.spawn_search(arg.to_string());
                }
            }
            "stats" => self.spawn_stats(),
            "ingest" => {
                if arg.is_empty() {
                    self.push_dim("用法：/ingest 目录");
                } else {
                    self.spawn_ingest(arg.to_string());
                }
            }
            "note" => self.save_note(arg),
            "alias" => self.spawn_alias(arg.to_string()),
            "memory" => self.spawn_memory(arg.to_string()),
            "experience" => self.spawn_experience(arg.to_string()),
            "adopt" | "good" => self.record_feedback("adopt", None, None),
            "reject" | "bad" => {
                let note = if arg.is_empty() {
                    None
                } else {
                    Some(arg.to_string())
                };
                self.record_feedback("reject", None, note)
            }
            "cite" | "open" => {
                if arg.is_empty() {
                    self.push_dim("用法：/cite 引用编号");
                } else {
                    self.record_feedback("cite", Some(arg.to_string()), None);
                }
            }
            "feedback" => self.spawn_feedback_list(),
            "inbox" => self.spawn_inbox_list(),
            "promote" => self.spawn_promote(arg.to_string()),
            "inbox-drop" => self.spawn_inbox_drop(arg.to_string()),
            "propose" => self.spawn_propose(arg.to_string()),
            "proposals" => self.spawn_proposals(),
            "apply" => self.spawn_apply(arg.to_string()),
            "discard" => self.spawn_reject(arg.to_string()),
            "why" => self.show_why(),
            "last" => self.show_last(),
            _ => self.push_dim(format!("未知命令：/{cmd}（/help 查看）")),
        }
    }

    fn help(&mut self) {
        for l in [
            "/search <关键词>   看双路召回结果",
            "/stats             索引概况",
            "/ingest <目录>     导入 Markdown",
            "/note <内容>       追加到收件箱（人写，非权威）",
            "/alias add 词 = 实体  学一个别名（冲突默认拒绝，--force 覆盖）",
            "/alias list / del 词",
            "/memory add 笔记路径 = 事实  新增记忆条（绑定来源）",
            "/memory list / rm 编号 / check / reextract",
            "/experience add 规则 / list / promote 编号 / reject 编号",
            "/adopt            上一轮答案有用（采纳）",
            "/reject [原因]    上一轮答案不对（纠正，触发记忆边 LTD）",
            "/cite 引用编号     打开并记录一次引用点击",
            "/feedback         最近反馈",
            "/inbox            查看收件箱",
            "/promote 编号 笔记路径  把收件箱条目提升进正式笔记",
            "/inbox-drop 编号  丢弃收件箱条目",
            "/propose 笔记路径 = 指令  生成 diff 提案（不写文件）",
            "/proposals        待处理提案",
            "/apply 编号       写入提案（唯一修改笔记的路径）",
            "/discard 编号     拒绝提案",
            "/last              重看上一轮检索",
            "/why               解释上一轮的名次和分数",
            "/clear             清屏",
            "/quit              退出，并按 精排→聊天→嵌入 关闭本机模型组件",
            "",
            "其它输入 = 提问：先检索笔记，再由模型带引用回答。",
        ] {
            self.push_dim(l);
        }
    }

    fn save_note(&mut self, text: &str) {
        if text.is_empty() {
            self.push_dim("用法：/note 内容");
            return;
        }
        let path = std::env::var("NOTE_INBOX").unwrap_or_else(|_| "inbox.md".into());
        let entry = format!(
            "\n## {}\n\n{}\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M"),
            text
        );
        let res = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| {
                use std::io::Write;
                f.write_all(entry.as_bytes())
            });
        match res {
            Ok(()) => self.push_dim(format!("已写入 {path}")),
            Err(e) => self.push_red(format!("写入失败: {e}")),
        }
    }

    fn spawn_search(&mut self, query: String) {
        let cfg = self.cfg.clone();
        let embedder = self.embedder.clone();
        let tx = self.tx.clone();
        let log = self.log.clone();
        self.busy = true;
        self.busy_label = "检索中…".into();
        self.status = None;
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                let (query_used, alias_hits) = aliases::expand(&client, &query)
                    .await
                    .unwrap_or_else(|_| (query.clone(), Vec::new()));
                let alpha = settings::get(&client, "alpha")
                    .await
                    .ok()
                    .flatten()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.05);
                let opts = SearchOptions {
                    fusion: Fusion::Rerank { alpha },
                    ..Default::default()
                };
                let outcome =
                    hybrid_search(&client, Some(embedder.as_ref()), &query_used, &opts).await?;
                Ok::<_, anyhow::Error>((outcome, alias_hits))
            }
            .await;

            match res {
                Ok((outcome, alias_hits)) => {
                    if !alias_hits.is_empty() {
                        let _ = tx.send(Msg::Lines(vec![Line::from(Span::styled(
                            format!(
                                "别名：{}",
                                alias_hits
                                    .iter()
                                    .map(|h| format!("{} → {}", h.alias, h.entity))
                                    .collect::<Vec<_>>()
                                    .join("，")
                            ),
                            Style::default().fg(Color::DarkGray),
                        ))]));
                    }
                    if let Some(log) = &log {
                        let _ = log.append(&json!({
                            "ts": chrono::Local::now().to_rfc3339(),
                            "kind": "search",
                            "query": query,
                            "query_used": outcome.query,
                            "aliases": alias_hits,
                            "candidates": {
                                "vector": outcome.candidates.vector,
                                "lexical": outcome.candidates.lexical
                            },
                            "elapsed_ms": outcome.elapsed_ms,
                            "hits": outcome.hits.iter().map(|h| json!({
                                "rank": h.rank,
                                "path": h.path,
                                "heading": h.heading,
                                "score": h.rrf,
                            })).collect::<Vec<_>>(),
                        }));
                    }
                    let _ = tx.send(Msg::SearchDone {
                        query,
                        outcome: Box::new(outcome),
                    });
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn spawn_stats(&mut self) {
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        self.busy = true;
        self.busy_label = "读取索引…".into();
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                let row = client
                    .query_one(
                        "select (select count(*) from notes),
                                (select count(*) from chunks),
                                (select count(*) from embeddings)",
                        &[],
                    )
                    .await?;
                let models = client
                    .query(
                        "select model, dim, count(*) from embeddings group by model, dim order by model",
                        &[],
                    )
                    .await?;
                Ok::<_, anyhow::Error>((row, models))
            }
            .await;

            match res {
                Ok((row, models)) => {
                    let mut lines = vec![Line::from(format!(
                        "笔记 {} 篇｜块 {}｜向量 {}",
                        row.get::<_, i64>(0),
                        row.get::<_, i64>(1),
                        row.get::<_, i64>(2)
                    ))];
                    for m in models {
                        lines.push(Line::from(Span::styled(
                            format!(
                                "  {}（{} 维）：{} 条",
                                m.get::<_, String>(0),
                                m.get::<_, i32>(1),
                                m.get::<_, i64>(2)
                            ),
                            Style::default().fg(Color::DarkGray),
                        )));
                    }
                    let _ = tx.send(Msg::Lines(lines));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn spawn_ingest(&mut self, dir: String) {
        let cfg = self.cfg.clone();
        let embedder = self.embedder.clone();
        let tx = self.tx.clone();
        self.busy = true;
        self.busy_label = "入库中…".into();
        self.status = None;
        let handle = tokio::spawn(async move {
            let res = async {
                let mut client = db::connect(&cfg.database_url).await?;
                db::init_schema(&client).await?;
                let path = std::path::PathBuf::from(&dir);
                crate::ingest::ingest_dir(
                    &mut client,
                    embedder.as_ref(),
                    &path,
                    Some(path.as_path()),
                    1200,
                )
                .await
            }
            .await;

            match res {
                Ok(stats) => {
                    let mut lines = vec![Line::from(format!(
                        "扫描 {} 个文件：写入 {} 篇（{} 块），未变 {} 篇，警告 {} 条",
                        stats.files_seen,
                        stats.notes_written,
                        stats.chunks_written,
                        stats.notes_skipped,
                        stats.warnings.len()
                    ))];
                    for w in stats.warnings.iter().take(5) {
                        lines.push(Line::from(Span::styled(
                            format!("  警告：{w}"),
                            Style::default().fg(Color::Gray),
                        )));
                    }
                    let _ = tx.send(Msg::Lines(lines));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn spawn_alias(&mut self, arg: String) {
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        self.busy = true;
        self.busy_label = "别名…".into();
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                db::init_schema(&client).await?;
                alias_command(&client, &arg).await
            }
            .await;
            match res {
                Ok(lines) => {
                    let _ = tx.send(Msg::Lines(lines));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn inbox_path() -> std::path::PathBuf {
        std::env::var("NOTE_INBOX")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("inbox.md"))
    }

    fn spawn_inbox_list(&mut self) {
        match inbox::list(&Self::inbox_path()) {
            Ok(entries) => {
                let mut lines = vec![Line::from(format!("收件箱 {} 条", entries.len()))];
                for e in entries {
                    let snip: String = e
                        .body
                        .chars()
                        .take(80)
                        .map(|c| if c == '\n' { ' ' } else { c })
                        .collect();
                    lines.push(Line::from(format!("#{} {}｜{}", e.index, e.title, snip)));
                }
                self.lines.extend(lines);
                self.scroll_back = 0;
            }
            Err(e) => self.push_red(format!("{e:#}")),
        }
    }

    fn spawn_inbox_drop(&mut self, arg: String) {
        let Ok(index) = arg.trim().parse::<usize>() else {
            self.push_dim("用法：/inbox-drop 编号");
            return;
        };
        match inbox::drop_entry(&Self::inbox_path(), index) {
            Ok(true) => self.push_dim(format!("已丢弃收件箱 #{index}")),
            Ok(false) => self.push_dim(format!("收件箱没有 #{index}")),
            Err(e) => self.push_red(format!("{e:#}")),
        }
    }

    fn spawn_promote(&mut self, arg: String) {
        let Some((idx, path)) = arg.trim().split_once(' ') else {
            self.push_dim("用法：/promote 编号 笔记路径");
            return;
        };
        let Ok(index) = idx.trim().parse::<usize>() else {
            self.push_dim("用法：/promote 编号 笔记路径");
            return;
        };
        let note_path = path.trim().to_string();
        if note_path.is_empty() {
            self.push_dim("用法：/promote 编号 笔记路径");
            return;
        }
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        self.busy = true;
        self.busy_label = "提升收件箱…".into();
        let inbox_path = Self::inbox_path();
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                db::init_schema(&client).await?;
                let root = settings::get(&client, "notes_root")
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("没有 notes_root；先 ingest 一次"))?;
                let target = std::path::PathBuf::from(root).join(&note_path);
                inbox::promote(&inbox_path, index, &target)?;
                Ok::<_, anyhow::Error>(target)
            }
            .await;
            match res {
                Ok(target) => {
                    let _ = tx.send(Msg::Lines(vec![Line::from(Span::styled(
                        format!(
                            "已提升 #{index} 到 {}；运行 /ingest 更新索引",
                            target.display()
                        ),
                        Style::default().fg(Color::White),
                    ))]));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn flush_extracts(&mut self) {
        if self.extract_queue.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.extract_queue);
        let cfg = self.cfg.clone();
        let chat = self.chat.clone();
        let ec = self.extract.clone();
        let tx = self.tx.clone();
        let log = self.log.clone();
        tokio::spawn(async move {
            let mut lines = Vec::new();
            for p in pending {
                let mut pointer = SessionPointer::new(p.session_id.clone());
                pointer.turn = p.turn;
                match extraction_after(
                    &cfg,
                    &chat,
                    &ec,
                    &p.question,
                    &p.answer,
                    &p.sources,
                    &pointer,
                    log.as_ref(),
                )
                .await
                {
                    Ok(ls) => lines.extend(ls),
                    Err(e) => lines.push(Line::from(Span::styled(
                        format!("提炼失败: {e:#}"),
                        Style::default().fg(Color::DarkGray),
                    ))),
                }
            }
            if !lines.is_empty() {
                let _ = tx.send(Msg::Lines(lines));
            }
        });
    }

    fn spawn_propose(&mut self, arg: String) {
        let Some((path, instruction)) = arg.split_once('=') else {
            self.push_dim("用法：/propose 笔记路径 = 指令");
            return;
        };
        let (path, instruction) = (path.trim().to_string(), instruction.trim().to_string());
        if path.is_empty() || instruction.is_empty() {
            self.push_dim("用法：/propose 笔记路径 = 指令");
            return;
        }
        let cfg = self.cfg.clone();
        let chat = self.chat.clone();
        let tx = self.tx.clone();
        let session_id = self.session.session_id.clone();
        let turn = self.session.turn as i32 + 1;
        self.busy = true;
        self.busy_label = "提案中…".into();
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                db::init_schema(&client).await?;
                propose::create(&client, &chat, &path, &instruction, &session_id, turn).await
            }
            .await;
            match res {
                Ok(p) => {
                    let mut lines = vec![Line::from(Span::styled(
                        format!("提案 #{}｜{}", p.id, p.note_path),
                        Style::default().fg(Color::White),
                    ))];
                    for l in p.diff.lines().take(120) {
                        let color = if l.starts_with('+') || l.starts_with('-') {
                            Color::White
                        } else if l.starts_with("@@") {
                            Color::Gray
                        } else {
                            Color::DarkGray
                        };
                        lines.push(Line::from(Span::styled(
                            l.to_string(),
                            Style::default().fg(color),
                        )));
                    }
                    let _ = tx.send(Msg::Lines(lines));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn spawn_proposals(&mut self) {
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        self.busy = true;
        self.busy_label = "提案…".into();
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                db::init_schema(&client).await?;
                propose::list(&client, 20).await
            }
            .await;
            match res {
                Ok(rows) => {
                    let mut lines = vec![Line::from(format!("待处理提案 {} 条", rows.len()))];
                    for (id, path, status) in rows {
                        lines.push(Line::from(format!("#{id} {status}｜{path}")));
                    }
                    let _ = tx.send(Msg::Lines(lines));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn spawn_apply(&mut self, arg: String) {
        let Ok(id) = arg.trim().parse::<i64>() else {
            self.push_dim("用法：/apply 提案编号");
            return;
        };
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        self.busy = true;
        self.busy_label = "应用提案…".into();
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                db::init_schema(&client).await?;
                propose::apply(&client, id).await
            }
            .await;
            match res {
                Ok(file) => {
                    let _ = tx.send(Msg::Lines(vec![Line::from(Span::styled(
                        format!("已写入 {}；运行 /ingest 更新索引", file.display()),
                        Style::default().fg(Color::White),
                    ))]));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn spawn_reject(&mut self, arg: String) {
        let Ok(id) = arg.trim().parse::<i64>() else {
            self.push_dim("用法：/discard 提案编号");
            return;
        };
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        self.busy = true;
        self.busy_label = "拒绝提案…".into();
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                db::init_schema(&client).await?;
                propose::reject(&client, id).await
            }
            .await;
            match res {
                Ok(true) => {
                    let _ = tx.send(Msg::Lines(vec![Line::from(format!("已拒绝 #{id}"))]));
                }
                Ok(false) => {
                    let _ = tx.send(Msg::Lines(vec![Line::from(format!(
                        "没有待处理提案：#{id}"
                    ))]));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn spawn_memory(&mut self, arg: String) {
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        self.busy = true;
        self.busy_label = "记忆…".into();
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                db::init_schema(&client).await?;
                memory_command(&client, &arg).await
            }
            .await;
            match res {
                Ok(lines) => {
                    let _ = tx.send(Msg::Lines(lines));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn record_feedback(&mut self, kind: &str, target: Option<String>, note: Option<String>) {
        let Some(ctx) = self.last_feedback.clone() else {
            self.push_dim("还没有可反馈的轮次");
            return;
        };

        if kind == "cite"
            && let Some(t) = &target
        {
            match t.parse::<usize>() {
                Ok(n) if n >= 1 && n <= ctx.sources.len() => {
                    let s = &ctx.sources[n - 1];
                    let snip: String = s
                        .text
                        .chars()
                        .take(140)
                        .map(|c| if c == '\n' { ' ' } else { c })
                        .collect();
                    self.push_dim(format!("引用 [{}] {}｜{}", n, s.head, snip));
                }
                _ => {
                    self.push_dim("引用编号超出范围");
                    return;
                }
            }
        }

        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        let log = self.log.clone();
        let kind = kind.to_string();
        self.busy = true;
        self.busy_label = "记录反馈…".into();
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                db::init_schema(&client).await?;
                let detail = json!({
                    "question": ctx.question,
                    "note": note,
                    "memory_ids": ctx.memory_ids,
                    "experience_candidates": ctx.experience_candidates,
                    "sources": ctx.sources.iter().enumerate().map(|(i, s)| json!({
                        "n": i + 1,
                        "chunk_id": s.chunk_id,
                        "path": s.path,
                        "heading": s.head,
                    })).collect::<Vec<_>>(),
                });
                let id = feedback::record(
                    &client,
                    Some(&ctx.session_id),
                    Some(ctx.turn as i32),
                    &kind,
                    target.as_deref(),
                    detail,
                )
                .await?;
                let mut promoted = Vec::new();
                if kind == "adopt" || kind == "cite" {
                    promoted = experience::reward(
                        &client,
                        &ctx.experience_candidates,
                        experience::promote_after(),
                    )
                    .await?;
                } else if kind == "reject" {
                    memory::penalize(&client, &ctx.memory_ids).await?;
                    experience::punish(&client, &ctx.experience_candidates).await?;
                }
                if let Some(log) = &log {
                    let _ = log.append(&json!({
                        "ts": chrono::Local::now().to_rfc3339(),
                        "kind": "feedback",
                        "session_id": ctx.session_id,
                        "turn": ctx.turn,
                        "feedback_id": id,
                        "feedback_kind": kind,
                        "experiences_promoted": promoted.len(),
                    }));
                }
                Ok::<_, anyhow::Error>((id, promoted))
            }
            .await;
            match res {
                Ok((id, promoted)) => {
                    let mut lines = vec![Line::from(Span::styled(
                        format!("反馈 #{id} 已记录：{kind}"),
                        Style::default().fg(Color::DarkGray),
                    ))];
                    for p in promoted {
                        lines.push(Line::from(Span::styled(
                            format!("经验晋升：{p}"),
                            Style::default().fg(Color::White),
                        )));
                    }
                    let _ = tx.send(Msg::Lines(lines));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn spawn_feedback_list(&mut self) {
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        self.busy = true;
        self.busy_label = "反馈…".into();
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                db::init_schema(&client).await?;
                feedback::list(&client, 10).await
            }
            .await;
            match res {
                Ok(rows) => {
                    let mut lines = vec![Line::from(format!("最近反馈 {} 条", rows.len()))];
                    for r in rows {
                        lines.push(Line::from(format!(
                            "#{} {} turn={} {} {}",
                            r.id,
                            r.created_at,
                            r.turn.unwrap_or_default(),
                            r.kind,
                            r.target.unwrap_or_default()
                        )));
                    }
                    let _ = tx.send(Msg::Lines(lines));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn spawn_experience(&mut self, arg: String) {
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        self.busy = true;
        self.busy_label = "经验…".into();
        let handle = tokio::spawn(async move {
            let res = async {
                let client = db::connect(&cfg.database_url).await?;
                db::init_schema(&client).await?;
                experience_command(&client, &arg).await
            }
            .await;
            match res {
                Ok(lines) => {
                    let _ = tx.send(Msg::Lines(lines));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                }
            }
            let _ = tx.send(Msg::Busy(false));
        });
        self.task = Some(handle);
    }

    fn spawn_chat(&mut self, question: String, allow_clarify: bool) {
        let cfg = self.cfg.clone();
        let embedder = self.embedder.clone();
        let chat = self.chat.clone();
        let rewrite = self.rewrite.clone();
        let select = self.select.clone();
        let extract_cfg = self.extract.clone();
        let verify_cfg = self.verify.clone();
        let rerank_cfg = self.rerank.clone();
        let tx = self.tx.clone();
        let log = self.log.clone();
        self.busy = true;
        self.busy_label = "判题中…".into();
        self.status = None;
        let pointer = self.session.clone();
        let handle = tokio::spawn(async move {
            let res = chat_turn(
                &cfg,
                embedder.as_ref(),
                &chat,
                &rewrite,
                &select,
                &verify_cfg,
                &rerank_cfg,
                allow_clarify,
                &question,
                &pointer,
                &tx,
                log.as_ref(),
            )
            .await;
            match res {
                Ok(turn) => {
                    let ChatTurn {
                        lines,
                        pointer,
                        clarify,
                        answer,
                        sources,
                        feedback,
                    } = turn;
                    let clarifying = clarify.is_some();
                    let _ = tx.send(Msg::ChatDone {
                        lines,
                        pointer: pointer.clone(),
                        feedback,
                    });
                    if let Some(req) = clarify {
                        let _ = tx.send(Msg::Clarify(req));
                    }
                    let _ = tx.send(Msg::Busy(false));

                    if !clarifying
                        && extract_cfg.enabled
                        && let Some(answer) = answer
                    {
                        match extract_cfg.mode {
                            ExtractMode::Immediate => {
                                let lines = extraction_after(
                                    &cfg,
                                    &chat,
                                    &extract_cfg,
                                    &question,
                                    &answer,
                                    &sources,
                                    &pointer,
                                    log.as_ref(),
                                )
                                .await
                                .unwrap_or_else(|e| {
                                    vec![Line::from(Span::styled(
                                        format!("提炼失败: {e:#}"),
                                        Style::default().fg(Color::DarkGray),
                                    ))]
                                });
                                if !lines.is_empty() {
                                    let _ = tx.send(Msg::Lines(lines));
                                }
                            }
                            _ => {
                                let _ = tx.send(Msg::QueueExtract(PendingExtract {
                                    question: question.clone(),
                                    answer,
                                    sources,
                                    session_id: pointer.session_id.clone(),
                                    turn: pointer.turn,
                                }));
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e:#}")));
                    let _ = tx.send(Msg::Busy(false));
                }
            }
        });
        self.task = Some(handle);
    }

    fn abort(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
        self.busy = false;
        self.status = None;
        self.busy_label.clear();
    }

    fn show_last(&mut self) {
        match &self.last_search {
            Some((query, outcome)) => {
                let mut lines = format_search(query, outcome);
                self.lines.append(&mut lines);
                self.scroll_back = 0;
            }
            None => self.push_dim("还没有检索记录"),
        }
    }

    fn show_why(&mut self) {
        match &self.last_search {
            Some((query, outcome)) => {
                let mut lines = vec![Line::from(Span::styled(
                    format!(
                        "「{query}」 向量 {} / 字面 {} 候选｜{:.3}s｜模型 {}",
                        outcome.candidates.vector,
                        outcome.candidates.lexical,
                        outcome.elapsed_ms as f64 / 1000.0,
                        outcome.embedder
                    ),
                    Style::default().fg(Color::White),
                ))];
                for h in &outcome.hits {
                    let mut meta = format!("score={:.4}", h.rrf);
                    if let Some(r) = h.vector_rank {
                        meta.push_str(&format!(
                            "  vec#{r}={:.3}",
                            h.vector_score.unwrap_or_default()
                        ));
                    }
                    if let Some(r) = h.lexical_rank {
                        meta.push_str(&format!(
                            "  lex#{r}={:.3}",
                            h.lexical_score.unwrap_or_default()
                        ));
                    }
                    let head = if h.heading.is_empty() {
                        h.path.clone()
                    } else {
                        format!("{} › {}", h.path, h.heading)
                    };
                    lines.push(Line::from(vec![
                        Span::styled(format!("[{}] ", h.rank), Style::default().fg(Color::Gray)),
                        Span::styled(head, Style::default().add_modifier(Modifier::BOLD)),
                        Span::styled(format!("  {meta}"), Style::default().fg(Color::DarkGray)),
                    ]));
                }
                self.lines.extend(lines);
                self.scroll_back = 0;
            }
            None => self.push_dim("还没有检索记录"),
        }
    }

    fn push_user(&mut self, text: &str) {
        self.lines.push(Line::from(vec![
            Span::styled(">>> ", Style::default().fg(Color::Gray)),
            Span::styled(
                text.to_string(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ]));
        self.scroll_back = 0;
    }

    fn push_dim(&mut self, text: impl Into<String>) {
        self.lines.push(Line::from(Span::styled(
            text.into(),
            Style::default().fg(Color::DarkGray),
        )));
        self.scroll_back = 0;
    }

    fn push_red(&mut self, text: impl Into<String>) {
        self.lines.push(Line::from(Span::styled(
            text.into(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )));
        self.scroll_back = 0;
    }

    fn draw(&mut self, f: &mut Frame) {
        let area = f.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(3),
                Constraint::Length(1),
                Constraint::Length(1),
            ])
            .split(area);

        let width = chunks[0].width.max(1) as usize;
        let total_visual: usize = self
            .lines
            .iter()
            .map(|line| {
                let w: usize = line
                    .spans
                    .iter()
                    .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
                    .sum();
                w.div_ceil(width).max(1)
            })
            .sum();
        let height = chunks[0].height as usize;
        let max_scroll = total_visual.saturating_sub(height);
        let scroll = max_scroll.saturating_sub(self.scroll_back.min(max_scroll));

        f.render_widget(
            Paragraph::new(Text::from(self.lines.clone()))
                .wrap(Wrap { trim: false })
                .style(Style::default().fg(Color::Gray))
                .scroll((scroll as u16, 0)),
            chunks[0],
        );

        let prefix = ">>> ";
        let text: String = self.input.iter().collect();
        let input_line = Line::from(vec![
            Span::styled(prefix, Style::default().fg(Color::Gray)),
            Span::styled(text, Style::default().fg(Color::White)),
        ]);
        f.render_widget(Paragraph::new(input_line), chunks[1]);
        let before_cursor: String = self.input[..self.cursor].iter().collect();
        let cx = chunks[1].x
            + UnicodeWidthStr::width(prefix) as u16
            + UnicodeWidthStr::width(before_cursor.as_str()) as u16;
        let cx = cx.min(chunks[1].x + chunks[1].width.saturating_sub(1));
        f.set_cursor_position((cx, chunks[1].y));

        // 输入 / 时的命令菜单：覆盖在输入行上方。
        let menu = self.menu_matches();
        if !menu.is_empty() {
            let idx = self.menu_index.min(menu.len() - 1);
            let max_rows = 8usize;
            let start = idx.saturating_sub(max_rows - 1);
            let end = (start + max_rows).min(menu.len());
            let rows = &menu[start..end];
            let height = rows.len() as u16;
            if height > 0 && chunks[0].height >= height {
                let rect = Rect {
                    x: chunks[1].x,
                    y: chunks[1].y.saturating_sub(height),
                    width: chunks[1].width,
                    height,
                };
                f.render_widget(Clear, rect);
                for (i, (name, desc)) in rows.iter().enumerate() {
                    let selected = start + i == idx;
                    let (cmd_style, desc_style) = if selected {
                        (
                            Style::default().add_modifier(Modifier::REVERSED | Modifier::BOLD),
                            Style::default().add_modifier(Modifier::REVERSED),
                        )
                    } else {
                        (
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD),
                            Style::default().fg(Color::Gray),
                        )
                    };
                    let line = Line::from(vec![
                        Span::styled(format!("{:<14}", name), cmd_style),
                        Span::styled(format!(" {desc}"), desc_style),
                    ]);
                    let row = Rect {
                        x: rect.x,
                        y: rect.y + i as u16,
                        width: rect.width,
                        height: 1,
                    };
                    f.render_widget(Paragraph::new(line).style(cmd_style), row);
                }
            }
        }

        self.draw_status(f, chunks[2]);
    }

    fn draw_status(&self, f: &mut Frame, area: Rect) {
        let (details, shown) = if let Some(s) = &self.status {
            (status_details(s), Some(s))
        } else if self.pending_clarify.is_some() {
            (
                "[等待别名确认] | 输入实体，回车跳过".to_string(),
                self.last_status.as_ref(),
            )
        } else if self.busy {
            (format!("[{}]", self.busy_label), self.last_status.as_ref())
        } else if let Some(s) = &self.last_status {
            (status_details(s), Some(s))
        } else {
            ("[就绪] | /help 查看命令".to_string(), None)
        };

        let (kv_ratio, kv_used, kv_ctx) = if let Some(s) = shown {
            let ctx = if s.kv_ctx > 0 {
                s.kv_ctx
            } else {
                self.chat.ctx
            };
            let used = if s.kv_used > 0 {
                s.kv_used
            } else {
                s.prompt_tokens + s.tokens as u32
            };
            let ratio = (used as f64 / ctx.max(1) as f64).clamp(0.0, 1.0);
            (ratio, used, ctx)
        } else {
            (0.0, 0, 0)
        };

        let (ctx_used, ctx_budget) = shown
            .map(|s| (s.context_used, s.context_budget))
            .unwrap_or((0, 0));
        let ctx_ratio = if ctx_budget > 0 {
            (ctx_used as f64 / ctx_budget as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let kv_suffix = if kv_ctx > 0 {
            format!(" {}/{}", comma(kv_used as u64), comma(kv_ctx as u64))
        } else {
            " --".to_string()
        };
        let ctx_suffix = if ctx_budget > 0 {
            format!(" {}/{}", comma(ctx_used as u64), comma(ctx_budget as u64))
        } else {
            " --".to_string()
        };

        let kv_label = " KV ";
        let ctx_label = " | 上下文 ";
        let fixed = UnicodeWidthStr::width(details.as_str())
            + UnicodeWidthStr::width(kv_label)
            + 2
            + UnicodeWidthStr::width(kv_suffix.as_str())
            + UnicodeWidthStr::width(ctx_label)
            + 2
            + UnicodeWidthStr::width(ctx_suffix.as_str());
        let free = (area.width as usize).saturating_sub(fixed);
        let kv_bar_w = (free / 2).clamp(4, 20);
        let ctx_bar_w = (free.saturating_sub(kv_bar_w)).clamp(4, 20);

        let mut spans = vec![Span::styled(details, Style::default().fg(Color::White))];
        spans.push(Span::styled(kv_label, Style::default().fg(Color::Gray)));
        spans.extend(bar_spans(kv_ratio, kv_bar_w));
        spans.push(Span::styled(kv_suffix, Style::default().fg(Color::Gray)));
        spans.push(Span::styled(ctx_label, Style::default().fg(Color::Gray)));
        spans.extend(bar_spans(ctx_ratio, ctx_bar_w));
        spans.push(Span::styled(ctx_suffix, Style::default().fg(Color::Gray)));

        f.render_widget(
            Paragraph::new(Line::from(spans)),
            Rect {
                x: area.x,
                y: area.y,
                width: area.width,
                height: 1,
            },
        );
    }
}

fn bar_spans(ratio: f64, width: usize) -> Vec<Span<'static>> {
    let units = (ratio * width as f64 * 2.0).round() as usize;
    let full = units / 2;
    let half = units % 2;
    let filled = format!("{}{}", "█".repeat(full), if half > 0 { "▌" } else { "" });
    vec![
        Span::styled("[", Style::default().fg(Color::Gray)),
        Span::styled(filled, Style::default().fg(Color::White)),
        Span::styled(
            " ".repeat(width - full - half),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled("]", Style::default().fg(Color::Gray)),
    ]
}

fn status_details(s: &StatusView) -> String {
    let phase = if s.finished {
        "完成"
    } else if s.tokens == 0 {
        "思考中"
    } else {
        "生成中"
    };
    format!(
        "[{} {:.1}s] | [{} token]",
        phase,
        s.elapsed_ms as f64 / 1000.0,
        comma(s.tokens)
    )
}

fn comma(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// 把模型返回的 Markdown 整段渲染成 TUI 行；配色只用白/灰。
fn render_markdown(md: &str) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut fence: Option<String> = None;

    for raw in md.lines() {
        let line = raw.trim_end();
        let trimmed = line.trim_start();

        if let Some(_lang) = &fence {
            if trimmed.starts_with("```") {
                out.push(Line::from(Span::styled(
                    "└────────",
                    Style::default().fg(Color::DarkGray),
                )));
                fence = None;
            } else {
                out.push(Line::from(vec![
                    Span::styled("│ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(line.to_string(), Style::default().fg(Color::Gray)),
                ]));
            }
            continue;
        }

        if trimmed.starts_with("```") {
            let lang = trimmed.trim_start_matches('`').trim().to_string();
            out.push(Line::from(Span::styled(
                if lang.is_empty() {
                    "┌".to_string()
                } else {
                    format!("┌ {lang}")
                },
                Style::default().fg(Color::DarkGray),
            )));
            fence = Some(lang);
            continue;
        }

        if let Some((_, text)) = heading(trimmed) {
            out.push(Line::from(Span::styled(
                text.to_string(),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }

        if is_rule(trimmed) {
            out.push(Line::from(Span::styled(
                "─".repeat(40),
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix('>') {
            let mut spans = vec![Span::styled("│ ", Style::default().fg(Color::DarkGray))];
            spans.extend(render_inline(
                rest.trim_start(),
                Style::default().fg(Color::Gray),
            ));
            out.push(Line::from(spans));
            continue;
        }

        if let Some((marker, rest)) = list_item(trimmed) {
            let mut spans = vec![Span::styled(marker, Style::default().fg(Color::Gray))];
            spans.extend(render_inline(rest, Style::default().fg(Color::Gray)));
            out.push(Line::from(spans));
            continue;
        }

        if trimmed.is_empty() {
            out.push(Line::from(""));
        } else {
            let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
            let mut spans = Vec::new();
            if !indent.is_empty() {
                spans.push(Span::styled(indent, Style::default()));
            }
            spans.extend(render_inline(trimmed, Style::default().fg(Color::Gray)));
            out.push(Line::from(spans));
        }
    }

    if fence.is_some() {
        out.push(Line::from(Span::styled(
            "└────────",
            Style::default().fg(Color::DarkGray),
        )));
    }
    out
}

fn heading(s: &str) -> Option<(usize, &str)> {
    let hashes = s.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &s[hashes..];
    rest.strip_prefix(' ').map(|r| (hashes, r.trim_start()))
}

fn is_rule(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    matches!(first, '-' | '*' | '_') && s.chars().count() >= 3 && chars.all(|c| c == first)
}

fn list_item(s: &str) -> Option<(String, &str)> {
    for p in ["- ", "* ", "+ "] {
        if let Some(rest) = s.strip_prefix(p) {
            return Some(("• ".to_string(), rest));
        }
    }
    let digits = s.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 && s[digits..].starts_with(". ") {
        return Some((format!("{}. ", &s[..digits]), &s[digits + 2..]));
    }
    None
}

fn render_inline(text: &str, base: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let mut spans = Vec::new();
    let mut buf = String::new();
    let mut i = 0usize;

    while i < chars.len() {
        let flush = |buf: &mut String, spans: &mut Vec<Span<'static>>| {
            if !buf.is_empty() {
                spans.push(Span::styled(std::mem::take(buf), base));
            }
        };

        if chars[i] == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
            let mut j = i + 2;
            while j + 1 < chars.len() && !(chars[j] == '*' && chars[j + 1] == '*') {
                j += 1;
            }
            if j + 1 < chars.len() {
                flush(&mut buf, &mut spans);
                let inner: String = chars[i + 2..j].iter().collect();
                spans.push(Span::styled(inner, base.add_modifier(Modifier::BOLD)));
                i = j + 2;
                continue;
            }
        }

        if chars[i] == '`'
            && let Some(off) = chars[i + 1..].iter().position(|c| *c == '`')
        {
            flush(&mut buf, &mut spans);
            let inner: String = chars[i + 1..i + 1 + off].iter().collect();
            spans.push(Span::styled(
                inner,
                Style::default().fg(Color::White).bg(Color::DarkGray),
            ));
            i = i + 1 + off + 1;
            continue;
        }

        if chars[i] == '*'
            && let Some(off) = chars[i + 1..].iter().position(|c| *c == '*')
        {
            flush(&mut buf, &mut spans);
            let inner: String = chars[i + 1..i + 1 + off].iter().collect();
            spans.push(Span::styled(inner, base.add_modifier(Modifier::ITALIC)));
            i = i + 1 + off + 1;
            continue;
        }

        if chars[i] == '['
            && let Some(cb) = chars[i + 1..].iter().position(|c| *c == ']')
        {
            let close = i + 1 + cb;
            if close + 1 < chars.len()
                && chars[close + 1] == '('
                && let Some(rp) = chars[close + 2..].iter().position(|c| *c == ')')
            {
                let end = close + 2 + rp;
                let label: String = chars[i + 1..close].iter().collect();
                let url: String = chars[close + 2..end].iter().collect();
                flush(&mut buf, &mut spans);
                spans.push(Span::styled(label, base.add_modifier(Modifier::UNDERLINED)));
                spans.push(Span::styled(
                    format!(" ({url})"),
                    Style::default().fg(Color::DarkGray),
                ));
                i = end + 1;
                continue;
            }
        }

        buf.push(chars[i]);
        i += 1;
    }

    if !buf.is_empty() {
        spans.push(Span::styled(buf, base));
    }
    spans
}

fn format_search(query: &str, out: &SearchOutcome) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        format!(
            "检索「{query}」 向量 {} / 字面 {} 候选｜{} ms",
            out.candidates.vector, out.candidates.lexical, out.elapsed_ms
        ),
        Style::default().fg(Color::White),
    ))];
    for w in &out.warnings {
        lines.push(Line::from(Span::styled(
            format!("注意：{w}"),
            Style::default().fg(Color::Gray),
        )));
    }
    if out.hits.is_empty() {
        lines.push(Line::from("没有命中。"));
    }
    for h in &out.hits {
        let head = if h.heading.is_empty() {
            h.path.clone()
        } else {
            format!("{} › {}", h.path, h.heading)
        };
        lines.push(Line::from(vec![
            Span::styled(format!("[{}] ", h.rank), Style::default().fg(Color::Gray)),
            Span::styled(head, Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("  score={:.4}", h.rrf),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        let snippet: String = h
            .text
            .chars()
            .take(160)
            .map(|c| if c == '\n' { ' ' } else { c })
            .collect();
        lines.push(Line::from(Span::styled(
            format!("    {snippet}"),
            Style::default().fg(Color::Gray),
        )));
        let mut meta = String::new();
        if let Some(r) = h.vector_rank {
            meta.push_str(&format!(
                "vec#{r} {:.3}  ",
                h.vector_score.unwrap_or_default()
            ));
        }
        if let Some(r) = h.lexical_rank {
            meta.push_str(&format!(
                "lex#{r} {:.3}",
                h.lexical_score.unwrap_or_default()
            ));
        }
        if !meta.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("    {meta}"),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }
    lines
}

#[derive(Clone, Copy, Default)]
struct ContextStat {
    budget: usize,
    used: usize,
    hits: usize,
    dropped: usize,
    memory_tokens: usize,
    memory_hits: usize,
    batches: usize,
}

/// 按模型给的相关性顺序装**全部**选中原文，不丢、不压。
fn collect_context(candidates: &[&Hit], order: &[usize]) -> Vec<(i64, String, String, String)> {
    order
        .iter()
        .filter_map(|&idx| {
            let h = candidates.get(idx)?;
            let head = if h.heading.is_empty() {
                h.path.clone()
            } else {
                format!("{} › {}", h.path, h.heading)
            };
            Some((h.chunk_id, h.path.clone(), head, h.text.clone()))
        })
        .collect()
}

/// 超过单批预算就拆批；单条超预算也单独成批，绝不截断。
fn split_batches(ctx: &[(i64, String, String, String)], budget: usize) -> Vec<Vec<usize>> {
    let budget = budget.max(64);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut used = 0usize;
    for (i, (_, _, head, text)) in ctx.iter().enumerate() {
        let cost = estimate_tokens(head) + 8 + estimate_tokens(text);
        if !cur.is_empty() && used + cost > budget {
            batches.push(std::mem::take(&mut cur));
            used = 0;
        }
        cur.push(i);
        used += cost;
    }
    if !cur.is_empty() {
        batches.push(cur);
    }
    batches
}

#[derive(Default)]
struct GenAgg {
    prompt_tokens: u32,
    completion_tokens: u64,
    ttft_ms: Option<u128>,
    total_ms: u128,
}

impl GenAgg {
    fn add(&mut self, r: &ChatResult) {
        self.prompt_tokens += r.prompt_tokens;
        self.completion_tokens += r.completion_tokens;
        self.total_ms += r.total_ms;
        if self.ttft_ms.is_none() {
            self.ttft_ms = r.ttft_ms;
        }
    }
}

fn context_text(ctx: &[(i64, String, String, String)], batch: &[usize]) -> String {
    let mut s = String::new();
    for &i in batch {
        let (_, _, head, text) = &ctx[i];
        s.push_str(&format!("[{}] {head}\n{text}\n\n", i + 1));
    }
    s
}

/// 判题查询与原问题合并：保留原关键词，再补改写词和指代对象。
fn merge_query(question: &str, rewritten: &str, resolved: Option<&ResolvedRef>) -> String {
    let mut q = question.to_string();
    let rw = rewritten.trim();
    if !rw.is_empty() && !q.contains(rw) {
        q.push(' ');
        q.push_str(rw);
    }
    if let Some(r) = resolved
        && !q.contains(&r.referent)
    {
        q.push(' ');
        q.push_str(&r.referent);
    }
    q
}

struct ChatTurn {
    lines: Vec<Line<'static>>,
    pointer: SessionPointer,
    clarify: Option<ClarifyRequest>,
    answer: Option<String>,
    sources: Vec<Source>,
    feedback: Option<FeedbackCtx>,
}

fn pick_unknown(unknown: &[String], alias_hits: &[AliasHit], question: &str) -> Option<String> {
    unknown
        .iter()
        .map(|s| s.trim())
        .find(|t| {
            !t.is_empty()
                && t.chars().count() <= 24
                && question.contains(*t)
                && !alias_hits.iter().any(|h| h.alias == *t)
        })
        .map(|t| t.to_string())
}

#[allow(clippy::too_many_arguments)]
async fn chat_turn(
    cfg: &Config,
    embedder: &dyn Embedder,
    chat_cfg: &ChatConfig,
    rewrite_cfg: &RewriteConfig,
    select_cfg: &SelectConfig,
    verify_cfg: &VerifyConfig,
    rerank_cfg: &RerankConfig,
    allow_clarify: bool,
    question: &str,
    pointer: &SessionPointer,
    tx: &UnboundedSender<Msg>,
    log: Option<&SessionLog>,
) -> Result<ChatTurn> {
    // 二层表 → 检索：先做确定性指代展开。
    let (pointer_query, resolved) = pointer.query_for(question);

    let mut lines = Vec::new();
    if let Some(r) = &resolved {
        lines.push(Line::from(Span::styled(
            format!("指针：{} → {}", r.mention, r.referent),
            Style::default().fg(Color::DarkGray),
        )));
    }

    // 判题改写：失败就退回原问题 + 指针。
    let mut rewrite_log = None;
    let mut unknown_terms: Vec<String> = Vec::new();
    let mut query_used = pointer_query.clone();
    if rewrite_cfg.enabled {
        let _ = tx.send(Msg::Phase("判题中…".into()));
        match rewrite::rewrite(chat_cfg, rewrite_cfg, question, pointer).await {
            Ok(r) => {
                query_used = merge_query(question, &r.data.query, resolved.as_ref());
                unknown_terms = r.data.unknown.clone();
                let qshort: String = r.data.query.chars().take(32).collect();
                let intent = if r.data.intent.is_empty() {
                    "-"
                } else {
                    r.data.intent.as_str()
                };
                lines.push(Line::from(Span::styled(
                    format!(
                        "判题 {:.1}s｜意图 {intent}｜查询 {qshort}",
                        r.elapsed_ms as f64 / 1000.0
                    ),
                    Style::default().fg(Color::DarkGray),
                )));
                rewrite_log = Some(json!({
                    "ok": true,
                    "query": r.data.query,
                    "intent": r.data.intent,
                    "entities": r.data.entities,
                    "unknown": r.data.unknown,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "elapsed_ms": r.elapsed_ms,
                }));
            }
            Err(e) => {
                lines.push(Line::from(Span::styled(
                    "判题失败，使用原问题",
                    Style::default().fg(Color::DarkGray),
                )));
                rewrite_log = Some(json!({"ok": false, "error": format!("{e:#}")}));
            }
        }
        let _ = tx.send(Msg::Phase("检索中…".into()));
    }

    let client = db::connect(&cfg.database_url).await?;
    let active_exp = experience::active(&client, 8).await.unwrap_or_default();

    // 别名展开：只追加实体，不改写原问题。
    let (expanded_query, alias_hits) = match aliases::expand(&client, &query_used).await {
        Ok(v) => v,
        Err(e) => {
            lines.push(Line::from(Span::styled(
                format!("别名展开失败: {e}"),
                Style::default().fg(Color::DarkGray),
            )));
            (query_used.clone(), Vec::new())
        }
    };
    if !alias_hits.is_empty() {
        lines.push(Line::from(Span::styled(
            format!(
                "别名：{}",
                alias_hits
                    .iter()
                    .map(|h| format!("{} → {}", h.alias, h.entity))
                    .collect::<Vec<_>>()
                    .join("，")
            ),
            Style::default().fg(Color::DarkGray),
        )));
    }
    query_used = expanded_query;

    // 反问：模型标了未知简称，而别名表里也没有，就先问再继续。
    if allow_clarify && let Some(term) = pick_unknown(&unknown_terms, &alias_hits, question) {
        lines.push(Line::from(Span::styled(
            format!("别名未命中：{term}"),
            Style::default().fg(Color::Gray),
        )));
        if let Some(log) = log {
            let _ = log.append(&json!({
                "ts": chrono::Local::now().to_rfc3339(),
                "kind": "clarify",
                "session_id": pointer.session_id,
                "turn": pointer.turn + 1,
                "question": question,
                "term": term,
                "query_used": query_used,
            }));
        }
        return Ok(ChatTurn {
            lines,
            pointer: pointer.clone(),
            clarify: Some(ClarifyRequest {
                question: question.to_string(),
                term,
            }),
            answer: None,
            sources: Vec::new(),
            feedback: None,
        });
    }

    // 候选经验只做相关匹配；是否晋升由后续采纳/引用/纠正事件决定。
    let experience_candidates: Vec<i64> = experience::match_candidates(&client, &query_used)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    let memories = memory::retrieve(&client, &query_used, 3)
        .await
        .unwrap_or_default();
    let memory_ids: Vec<i64> = memories.iter().map(|m| m.id).collect();
    if memory_ids.len() >= 2 {
        let _ = memory::strengthen(&client, &memory_ids).await;
    }
    let mut mem_block = String::new();
    if !memories.is_empty() {
        mem_block.push_str("记忆（可能过期，冲突时以笔记片段为准）：\n");
        for m in &memories {
            mem_block.push_str(&format!(
                "- {}（来源：{}）\n",
                m.text,
                m.note_path.as_deref().unwrap_or("?")
            ));
        }
    }
    let mem_tokens = estimate_tokens(&mem_block);
    if !memories.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("记忆：{} 条（{} token）", memories.len(), mem_tokens),
            Style::default().fg(Color::DarkGray),
        )));
    }

    let alpha = settings::get(&client, "alpha")
        .await
        .ok()
        .flatten()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.05);
    let mut outcome = hybrid_search(
        &client,
        Some(embedder),
        &query_used,
        &SearchOptions {
            k: 20,
            fusion: Fusion::Rerank { alpha },
            ..Default::default()
        },
    )
    .await?;

    // 第 1.5 遍：交叉编码器精排。失败就保留原顺序。
    let mut rerank_log = None;
    if rerank_cfg.enabled && !outcome.hits.is_empty() {
        let docs: Vec<String> = outcome
            .hits
            .iter()
            .map(|h| {
                let text: String = h.text.chars().take(400).collect();
                format!("{}\n{}", h.heading, text)
            })
            .collect();
        match rerank::rerank(rerank_cfg, &query_used, &docs).await {
            Ok((scores, ms)) => {
                let mut old = std::mem::take(&mut outcome.hits);
                let mut new_hits = Vec::with_capacity(old.len());
                for (idx, _) in &scores {
                    if let Some(h) = old.get(*idx) {
                        new_hits.push(h.clone());
                    }
                }
                for h in old.drain(..) {
                    if !new_hits.iter().any(|x| x.chunk_id == h.chunk_id) {
                        new_hits.push(h);
                    }
                }
                outcome.hits = new_hits;
                lines.push(Line::from(Span::styled(
                    format!("精排 {ms}ms｜{} 条", docs.len()),
                    Style::default().fg(Color::DarkGray),
                )));
                rerank_log = Some(json!({
                    "ok": true,
                    "scored": docs.len(),
                    "elapsed_ms": ms,
                }));
            }
            Err(e) => {
                lines.push(Line::from(Span::styled(
                    format!("精排失败，保留原顺序：{e:#}"),
                    Style::default().fg(Color::Gray),
                )));
                rerank_log = Some(json!({"ok": false, "error": format!("{e:#}")}));
            }
        }
    }

    let hit_meta: Vec<(i64, String, String)> = outcome
        .hits
        .iter()
        .map(|h| (h.chunk_id, h.path.clone(), h.heading.clone()))
        .collect();
    let mut pointer_after = pointer.clone();
    pointer_after.advance(question, &hit_meta, resolved.clone());

    for w in &outcome.warnings {
        lines.push(Line::from(Span::styled(
            format!("注意：{w}"),
            Style::default().fg(Color::Gray),
        )));
    }

    let candidates: Vec<&Hit> = outcome.hits.iter().take(20).collect();
    let mut select_log = None;
    let mut order: Vec<usize> = (0..candidates.len().min(6)).collect();

    if candidates.is_empty() {
        lines.push(Line::from("没有检索到相关笔记，先 /ingest 导入一些。"));
        log_turn(
            log,
            pointer,
            &pointer_after,
            question,
            &query_used,
            resolved,
            &outcome,
            None,
            ContextStat {
                budget: chat_cfg.context_budget,
                memory_tokens: mem_tokens,
                memory_hits: memories.len(),
                ..Default::default()
            },
            rewrite_log,
            None,
            alias_hits,
            memories,
            active_exp,
            None,
            None,
        );
        return Ok(ChatTurn {
            lines,
            pointer: pointer_after,
            clarify: None,
            answer: None,
            sources: Vec::new(),
            feedback: None,
        });
    }

    // 第 2 遍：模型只选编号，程序按预算装原文。
    if select_cfg.enabled {
        let _ = tx.send(Msg::Phase("选材中…".into()));
        match select::select(chat_cfg, select_cfg, question, &candidates).await {
            Ok(r) => {
                let too_few = r.picks.len() < 2;
                if too_few {
                    order = (0..candidates.len().min(6)).collect();
                    lines.push(Line::from(Span::styled(
                        "选材少于 2 条，回退检索名次",
                        Style::default().fg(Color::Gray),
                    )));
                } else {
                    order = r.picks.clone();
                }
                lines.push(Line::from(Span::styled(
                    format!(
                        "选材 {:.1}s｜选 {}/{} 条",
                        r.elapsed_ms as f64 / 1000.0,
                        r.picks.len(),
                        candidates.len()
                    ),
                    Style::default().fg(Color::DarkGray),
                )));
                select_log = Some(json!({
                    "ok": true,
                    "pick": r.picks,
                    "fallback": too_few,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "elapsed_ms": r.elapsed_ms,
                }));
            }
            Err(e) => {
                lines.push(Line::from(Span::styled(
                    "选材失败，退回检索名次",
                    Style::default().fg(Color::DarkGray),
                )));
                select_log = Some(json!({"ok": false, "error": format!("{e:#}")}));
            }
        }
        let _ = tx.send(Msg::Phase("生成中…".into()));
    }

    // 恒定 skill 保持逐字节稳定；经验放动态块，避免破坏 prefix cache。
    let system = SYSTEM_PROMPT.to_string();
    let exp_block = if active_exp.is_empty() {
        String::new()
    } else {
        format!(
            "已晋升经验（按此执行）：\n- {}\n\n",
            active_exp.join("\n- ")
        )
    };
    let mem_prefix = format!("{exp_block}{mem_block}");

    // 给 prompt 固定开销和输出留出余量，再决定每批片段预算。
    let overhead = estimate_tokens(&system)
        + estimate_tokens(question)
        + estimate_tokens(&exp_block)
        + estimate_tokens(&mem_block)
        + 64;
    let available = (chat_cfg.ctx as usize).saturating_sub(overhead + chat_cfg.max_tokens as usize);
    let snippet_budget = chat_cfg.context_budget.min(available).max(64);
    let all_ctx = collect_context(&candidates, &order);
    if all_ctx.is_empty() {
        lines.push(Line::from("没有可用片段。"));
        log_turn(
            log,
            pointer,
            &pointer_after,
            question,
            &query_used,
            resolved,
            &outcome,
            None,
            ContextStat {
                budget: snippet_budget,
                memory_tokens: mem_tokens,
                memory_hits: memories.len(),
                ..Default::default()
            },
            rewrite_log,
            select_log,
            alias_hits,
            memories,
            active_exp,
            None,
            None,
        );
        return Ok(ChatTurn {
            lines,
            pointer: pointer_after,
            clarify: None,
            answer: None,
            sources: Vec::new(),
            feedback: None,
        });
    }

    let batches = split_batches(&all_ctx, snippet_budget);
    let total_tokens: usize = all_ctx
        .iter()
        .map(|(_, _, head, text)| estimate_tokens(head) + 8 + estimate_tokens(text))
        .sum();
    let ctx_stat = ContextStat {
        budget: snippet_budget,
        used: total_tokens,
        hits: all_ctx.len(),
        dropped: 0,
        memory_tokens: mem_tokens,
        memory_hits: memories.len(),
        batches: batches.len(),
    };

    let sources: Vec<Source> = all_ctx
        .iter()
        .map(|(chunk_id, path, head, text)| Source {
            chunk_id: *chunk_id,
            path: path.clone(),
            head: head.clone(),
            text: text.clone(),
        })
        .collect();

    let res = if batches.len() <= 1 {
        let batch = batches.first().cloned().unwrap_or_default();
        let context = context_text(&all_ctx, &batch);
        let messages = vec![
            json!({"role": "system", "content": system}),
            json!({
                "role": "user",
                "content": format!("{mem_prefix}笔记片段：\n\n{context}\n问题：{question}")
            }),
        ];
        let tx2 = tx.clone();
        let progress = move |mut p: ChatProgress| {
            p.context_used = ctx_stat.used as u32;
            p.context_budget = ctx_stat.budget as u32;
            let _ = tx2.send(Msg::Status(p));
        };
        stream_chat(chat_cfg, messages, progress).await?
    } else {
        // 不压缩：分批各答，再做一次整合。
        let mut agg = GenAgg::default();
        let mut partials = Vec::new();
        let total_batches = batches.len();
        for (i, batch) in batches.iter().enumerate() {
            let _ = tx.send(Msg::Phase(format!("分批 {}/{}", i + 1, total_batches)));
            let context = context_text(&all_ctx, batch);
            let messages = vec![
                json!({"role": "system", "content": system}),
                json!({
                    "role": "user",
                    "content": format!(
                        "{mem_prefix}这是第 {}/{} 批笔记片段。只依据本批片段回答，引用保留全局编号。\n\n笔记片段：\n\n{context}\n问题：{question}",
                        i + 1,
                        total_batches
                    )
                }),
            ];
            let tx2 = tx.clone();
            let progress = move |mut p: ChatProgress| {
                p.context_used = ctx_stat.used as u32;
                p.context_budget = ctx_stat.budget as u32;
                let _ = tx2.send(Msg::Status(p));
            };
            let r = stream_chat(chat_cfg, messages, progress).await?;
            agg.add(&r);
            partials.push(r.content);
        }

        let _ = tx.send(Msg::Phase("整合分批答案".into()));
        let user = format!(
            "{exp_block}问题：{question}\n\n分批答案：\n{}\n\n请整合为一个最终答案，保留 [n] 引用，不要引入新事实。",
            partials.join("\n\n---\n\n")
        );
        let messages = vec![
            json!({"role": "system", "content": system}),
            json!({"role": "user", "content": user}),
        ];
        let tx2 = tx.clone();
        let progress = move |mut p: ChatProgress| {
            p.context_used = ctx_stat.used as u32;
            p.context_budget = ctx_stat.budget as u32;
            let _ = tx2.send(Msg::Status(p));
        };
        let r = stream_chat(chat_cfg, messages, progress).await?;
        agg.add(&r);
        ChatResult {
            content: r.content,
            prompt_tokens: agg.prompt_tokens,
            completion_tokens: agg.completion_tokens,
            ttft_ms: agg.ttft_ms,
            total_ms: agg.total_ms,
        }
    };

    let verify_res = verify::verify(chat_cfg, verify_cfg, &res.content, &sources).await;
    let refused = verify_cfg.strict && !verify_res.ok();
    if refused {
        lines.push(Line::from(Span::styled(
            "据现有笔记无法可靠回答（引用校验未通过）。",
            Style::default().fg(Color::Gray),
        )));
    } else {
        lines.extend(render_markdown(&res.content));
        if res.content.trim().is_empty() {
            lines.push(Line::from(Span::styled(
                "（模型没有输出内容）",
                Style::default().fg(Color::White),
            )));
        }
    }
    if !verify_res.ok() && !refused {
        let reason = if verify_res.reason.is_empty() {
            "未通过".to_string()
        } else {
            verify_res.reason.clone()
        };
        lines.push(Line::from(Span::styled(
            format!("校验：{reason}"),
            Style::default().fg(Color::Gray),
        )));
    }
    let verify_log = json!({
        "cited": verify_res.cited,
        "invalid": verify_res.invalid,
        "supported": verify_res.supported,
        "reason": verify_res.reason,
        "strict": verify_cfg.strict,
    });
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "来源",
        Style::default().fg(Color::DarkGray),
    )));
    for (i, (_, _, head, _)) in all_ctx.iter().enumerate() {
        lines.push(Line::from(Span::styled(
            format!("[{}] {head}", i + 1),
            Style::default().fg(Color::DarkGray),
        )));
    }
    let ttft = res
        .ttft_ms
        .map(|m| format!("{:.1}s", m as f64 / 1000.0))
        .unwrap_or_else(|| "-".to_string());
    let ctx_note = if ctx_stat.batches > 1 {
        format!(
            "上下文 {} token（{} 条，分 {} 批）",
            ctx_stat.used, ctx_stat.hits, ctx_stat.batches
        )
    } else {
        format!(
            "上下文 {}/{} token（{} 条）",
            ctx_stat.used, ctx_stat.budget, ctx_stat.hits
        )
    };
    lines.push(Line::from(Span::styled(
        format!(
            "完成 {:.1}s｜首 token {ttft}｜{} token｜{}",
            res.total_ms as f64 / 1000.0,
            res.completion_tokens,
            ctx_note
        ),
        Style::default().fg(Color::DarkGray),
    )));

    log_turn(
        log,
        pointer,
        &pointer_after,
        question,
        &query_used,
        resolved,
        &outcome,
        Some(&res),
        ctx_stat,
        rewrite_log,
        select_log,
        alias_hits,
        memories,
        active_exp,
        Some(verify_log),
        rerank_log,
    );

    let feedback = Some(FeedbackCtx {
        session_id: pointer_after.session_id.clone(),
        turn: pointer_after.turn,
        question: question.to_string(),
        sources: sources.clone(),
        memory_ids,
        experience_candidates,
    });

    Ok(ChatTurn {
        lines,
        pointer: pointer_after,
        clarify: None,
        answer: Some(res.content.clone()),
        sources,
        feedback,
    })
}

#[allow(clippy::too_many_arguments)]
async fn extraction_after(
    cfg: &Config,
    chat_cfg: &ChatConfig,
    ec: &ExtractConfig,
    question: &str,
    answer: &str,
    sources: &[Source],
    pointer: &SessionPointer,
    log: Option<&SessionLog>,
) -> Result<Vec<Line<'static>>> {
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    let client = db::connect(&cfg.database_url).await?;
    let raw = extract::extract(chat_cfg, ec, question, answer, sources).await?;
    let delta_id = extract::enqueue(
        &client,
        raw,
        sources,
        &pointer.session_id,
        pointer.turn as i32,
    )
    .await?;
    let report = extract::merge_one(&client, delta_id, ec.max_memories, ec.max_aliases).await?;

    if let Some(log) = log {
        let _ = log.append(&json!({
            "ts": chrono::Local::now().to_rfc3339(),
            "kind": "extract",
            "session_id": pointer.session_id,
            "turn": pointer.turn,
            "delta_id": report.delta_id,
            "memories_added": report.memories_added,
            "memories_duplicate": report.memories_duplicate,
            "aliases_added": report.aliases_added,
            "aliases_skipped": report.aliases_skipped,
            "experiences_added": report.experiences_added,
            "experiences_duplicate": report.experiences_duplicate,
            "errors": report.errors,
        }));
    }

    let mut lines = vec![Line::from(Span::styled(
        format!(
            "提炼 #{}｜记忆 +{}（重复 {}）｜别名 +{}（跳过 {}）｜经验 +{} 候选（重复 {}）",
            report.delta_id,
            report.memories_added,
            report.memories_duplicate,
            report.aliases_added,
            report.aliases_skipped,
            report.experiences_added,
            report.experiences_duplicate
        ),
        Style::default().fg(Color::DarkGray),
    ))];
    for e in report.errors.iter().take(3) {
        lines.push(Line::from(Span::styled(
            format!("  提炼警告: {e}"),
            Style::default().fg(Color::Gray),
        )));
    }
    Ok(lines)
}

async fn alias_command(client: &tokio_postgres::Client, arg: &str) -> Result<Vec<Line<'static>>> {
    let arg = arg.trim();
    if arg.is_empty() || arg == "list" {
        let rows = aliases::list(client, None).await?;
        let mut lines = vec![Line::from(format!("别名 {} 条", rows.len()))];
        for r in rows {
            lines.push(Line::from(format!("{} → {}", r.alias, r.entity)));
        }
        return Ok(lines);
    }
    if let Some(rest) = arg.strip_prefix("add ") {
        let force = rest.contains("--force");
        let rest = rest.replace("--force", "");
        let Some((a, e)) = rest.split_once('=') else {
            return Ok(vec![Line::from("用法：/alias add 别名 = 实体 [--force]")]);
        };
        let (a, e) = (a.trim(), e.trim());
        let msg = match aliases::add(client, a, e, force).await? {
            AddOutcome::Created => format!("已新增：{a} → {e}"),
            AddOutcome::Updated => format!("已更新：{a} → {e}"),
            AddOutcome::Unchanged => format!("未变化：{a} → {e}"),
        };
        return Ok(vec![Line::from(msg)]);
    }
    if let Some(rest) = arg.strip_prefix("del ").or_else(|| arg.strip_prefix("rm ")) {
        let alias = rest.trim();
        return Ok(vec![Line::from(if aliases::remove(client, alias).await? {
            format!("已删除：{alias}")
        } else {
            format!("没有这个别名：{alias}")
        })]);
    }
    Ok(vec![Line::from(
        "用法：/alias list | /alias add 别名 = 实体 [--force] | /alias del 别名",
    )])
}

async fn experience_command(
    client: &tokio_postgres::Client,
    arg: &str,
) -> Result<Vec<Line<'static>>> {
    let arg = arg.trim();
    if arg.is_empty() || arg == "list" {
        let rows = experience::list(client).await?;
        let mut lines = vec![Line::from(format!("经验 {} 条", rows.len()))];
        for r in rows {
            lines.push(Line::from(format!(
                "#{} {} uses={}：{}",
                r.id, r.status, r.uses, r.text
            )));
        }
        return Ok(lines);
    }
    if let Some(text) = arg.strip_prefix("add ") {
        let msg = match experience::add_candidate(client, text.trim(), "manual", 0).await? {
            experience::AddOutcome::Created(id) => format!("已新增经验候选 #{id}"),
            experience::AddOutcome::Existing(id) => format!("已存在经验 #{id}"),
        };
        return Ok(vec![Line::from(msg)]);
    }
    if let Some(rest) = arg.strip_prefix("promote ") {
        let id: i64 = rest
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("编号不是数字"))?;
        return Ok(vec![Line::from(
            if experience::promote(client, id).await? {
                format!("已晋升 #{id}")
            } else {
                format!("没有可晋升的候选：#{id}")
            },
        )]);
    }
    if let Some(rest) = arg.strip_prefix("reject ") {
        let id: i64 = rest
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("编号不是数字"))?;
        return Ok(vec![Line::from(if experience::reject(client, id).await? {
            format!("已拒绝 #{id}")
        } else {
            format!("没有这条经验：#{id}")
        })]);
    }
    Ok(vec![Line::from(
        "用法：/experience list | add 规则 | promote 编号 | reject 编号",
    )])
}

async fn memory_command(client: &tokio_postgres::Client, arg: &str) -> Result<Vec<Line<'static>>> {
    let arg = arg.trim();
    if arg.is_empty() || arg == "list" {
        let rows = memory::list(client).await?;
        let mut lines = vec![Line::from(format!("记忆 {} 条", rows.len()))];
        for r in rows {
            let stale = if r.status == "stale" { " [stale]" } else { "" };
            lines.push(Line::from(format!(
                "#{} {}{}：{}",
                r.id, r.status, stale, r.text
            )));
        }
        return Ok(lines);
    }
    if let Some(rest) = arg.strip_prefix("add ") {
        let pin = rest.contains("--pin");
        let rest = rest.replace("--pin", "");
        let Some((note, text)) = rest.split_once('=') else {
            return Ok(vec![Line::from(
                "用法：/memory add 笔记路径 = 事实 [--pin]",
            )]);
        };
        let msg = match memory::add(client, text.trim(), note.trim(), None, pin).await? {
            memory::AddOutcome::Created(id) => format!("已新增记忆 #{id}"),
            memory::AddOutcome::Existing(id) => format!("已存在记忆 #{id}"),
        };
        return Ok(vec![Line::from(msg)]);
    }
    if let Some(rest) = arg.strip_prefix("rm ").or_else(|| arg.strip_prefix("del ")) {
        let id: i64 = rest
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("编号不是数字"))?;
        return Ok(vec![Line::from(if memory::remove(client, id).await? {
            format!("已软删除 #{id}")
        } else {
            format!("没有这个记忆：#{id}")
        })]);
    }
    if arg == "check" {
        let n = memory::refresh(client).await?;
        return Ok(vec![Line::from(format!("检查完成，降级 {n} 条"))]);
    }
    if arg == "reextract" {
        let (revived, left) = memory::reextract(client, 100).await?;
        return Ok(vec![Line::from(format!(
            "重提炼：恢复 {revived} 条，仍失效 {left} 条"
        ))]);
    }
    Ok(vec![Line::from(
        "用法：/memory list | add 笔记路径 = 事实 [--pin] | rm 编号 | check | reextract",
    )])
}

#[allow(clippy::too_many_arguments)]
fn log_turn(
    log: Option<&SessionLog>,
    before: &SessionPointer,
    after: &SessionPointer,
    question: &str,
    query_used: &str,
    resolved: Option<ResolvedRef>,
    outcome: &SearchOutcome,
    answer: Option<&ChatResult>,
    ctx: ContextStat,
    rewrite: Option<serde_json::Value>,
    select: Option<serde_json::Value>,
    aliases: Vec<AliasHit>,
    memories: Vec<MemoryHit>,
    experiences: Vec<String>,
    verify: Option<serde_json::Value>,
    rerank: Option<serde_json::Value>,
) {
    let Some(log) = log else {
        return;
    };
    let _ = log.append(&json!({
        "ts": chrono::Local::now().to_rfc3339(),
        "kind": "turn",
        "session_id": after.session_id,
        "turn": after.turn,
        "question": question,
        "query_used": query_used,
        "pointer_before": before,
        "resolved": resolved,
        "retrieval": {
            "candidates": {
                "vector": outcome.candidates.vector,
                "lexical": outcome.candidates.lexical,
            },
            "elapsed_ms": outcome.elapsed_ms,
            "hits": outcome.hits.iter().map(|h| json!({
                "chunk_id": h.chunk_id,
                "path": h.path,
                "heading": h.heading,
                "score": h.rrf,
                "vector_rank": h.vector_rank,
                "vector_score": h.vector_score,
                "lexical_rank": h.lexical_rank,
                "lexical_score": h.lexical_score,
            })).collect::<Vec<_>>(),
        },
        "answer": answer.map(|r| json!({
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "ttft_ms": r.ttft_ms,
            "total_ms": r.total_ms,
        })),
        "context": {
            "budget": ctx.budget,
            "estimated_tokens": ctx.used,
            "hits": ctx.hits,
            "dropped": ctx.dropped,
            "memory_tokens": ctx.memory_tokens,
            "memory_hits": ctx.memory_hits,
        },
        "memories": memories,
        "experiences": experiences,
        "verify": verify,
        "rerank": rerank,
        "rewrite": rewrite,
        "select": select,
        "aliases": aliases,
        "pointer_after": after,
    }));
}

#[cfg(test)]
mod md_tests {
    use super::*;

    fn text(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn heading_bold_and_code() {
        let out = render_markdown("# 标题\n**粗体** 和 `代码`");
        assert_eq!(text(&out[0]), "标题");
        assert_eq!(
            out[0].spans[0].style,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD)
        );
        assert_eq!(text(&out[1]), "粗体 和 代码");
        assert_eq!(
            out[1].spans[0].style,
            Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::BOLD)
        );
        assert_eq!(
            out[1].spans[2].style,
            Style::default().fg(Color::White).bg(Color::DarkGray)
        );
    }

    #[test]
    fn lists_quotes_and_fence() {
        let out = render_markdown("- 一\n> 引用\n```rust\nlet x = 1;\n```");
        assert_eq!(text(&out[0]), "• 一");
        assert_eq!(text(&out[1]), "│ 引用");
        assert_eq!(text(&out[2]), "┌ rust");
        assert_eq!(text(&out[3]), "│ let x = 1;");
        assert_eq!(text(&out[4]), "└────────");
    }

    #[test]
    fn link_and_rule() {
        let out = render_markdown("见 [文档](https://example.com)\n---");
        assert_eq!(text(&out[0]), "见 文档 (https://example.com)");
        assert_eq!(text(&out[1]), "─".repeat(40));
    }
}
