use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use chrono::Local;
use tokio_postgres::NoTls;

#[derive(Debug, Clone)]
pub struct Component {
    pub id: &'static str,
    pub label: &'static str,
    pub port: u16,
    pub script: &'static str,
}

#[derive(Debug, Clone)]
pub struct SupervisorConfig {
    pub log_dir: PathBuf,
    pub database_url: String,
    pub pgdata: PathBuf,
    pub pg_ctl: PathBuf,
    pub components: Vec<Component>,
}

impl SupervisorConfig {
    pub fn new(
        log_dir: PathBuf,
        database_url: String,
        want_embed: bool,
        want_chat: bool,
        want_rerank: bool,
    ) -> Self {
        let mut components = Vec::new();
        if want_embed {
            components.push(Component {
                id: "embed",
                label: "嵌入服务",
                port: port_from_env("NOTE_EMBED_URL", 8081),
                script: "start-embed-server.sh",
            });
        }
        if want_chat {
            components.push(Component {
                id: "chat",
                label: "聊天服务",
                port: port_from_env("NOTE_CHAT_URL", 8082),
                script: "start-chat-server.sh",
            });
        }
        if want_rerank {
            components.push(Component {
                id: "rerank",
                label: "精排服务",
                port: port_from_env("NOTE_RERANK_URL", 8083),
                script: "start-rerank-server.sh",
            });
        }
        Self {
            log_dir,
            database_url,
            pgdata: std::env::var("NOTE_PGDATA")
                .map(PathBuf::from)
                .or_else(|_| std::env::var("HOME").map(|home| PathBuf::from(home).join("pgdata")))
                .unwrap_or_else(|_| PathBuf::from("/var/lib/pgdata")),
            pg_ctl: std::env::var("NOTE_PG_CTL")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/usr/lib/postgresql/18/bin/pg_ctl")),
            components,
        }
    }
}

/// URL 是否指向本机（只有本机组件才由看门狗拉起）。
pub fn env_url_is_local(key: &str, default: &str) -> bool {
    let url = std::env::var(key).unwrap_or_else(|_| default.to_string());
    url.contains("127.0.0.1") || url.contains("localhost")
}

fn port_from_env(key: &str, default: u16) -> u16 {
    let url = std::env::var(key).unwrap_or_default();
    url.rsplit(':')
        .next()
        .and_then(|p| p.split('/').next())
        .and_then(|p| p.parse().ok())
        .unwrap_or(default)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Health {
    Up,
    Loading,
    Down,
    Busy,
}

/// 四段式日志：[时间] [级别] [动作] [结果]
pub fn log(sc: &SupervisorConfig, level: &str, action: impl AsRef<str>, result: impl AsRef<str>) {
    fn clean(s: &str) -> String {
        s.replace(['\r', '\n', ']'], " ")
    }
    let line = format!(
        "[{}] [{}] [{}] [{}]",
        Local::now().format("%Y-%m-%d %H:%M:%S"),
        level,
        clean(action.as_ref()),
        clean(result.as_ref())
    );
    println!("{line}");
    let path = sc.log_dir.join("supervisor.log");
    let _ = fs::create_dir_all(&sc.log_dir);
    if fs::metadata(&path)
        .map(|m| m.len() > 2 * 1024 * 1024)
        .unwrap_or(false)
    {
        let _ = fs::rename(&path, sc.log_dir.join("supervisor.log.1"));
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{line}");
    }
}

pub async fn ensure_all(sc: &SupervisorConfig, verbose: bool) -> bool {
    log(
        sc,
        "INFO",
        "开始自检",
        format!("{} 个组件", sc.components.len() + 1),
    );
    let mut ok = ensure_postgres(sc, verbose).await;
    for c in &sc.components {
        ok &= ensure_component(sc, c, verbose).await;
    }
    log(
        sc,
        if ok { "INFO" } else { "ERROR" },
        "自检结束",
        if ok {
            "全部就绪"
        } else {
            "存在未就绪组件"
        },
    );
    ok
}

/// 按启动的逆序关闭本机组件：精排 → 聊天 → 嵌入；PostgreSQL 不动。
pub async fn shutdown_all(sc: &SupervisorConfig) {
    let mut components = sc.components.clone();
    components.reverse();
    for c in &components {
        shutdown_component(sc, c, 10).await;
    }
    log(sc, "INFO", "关闭完成", "本机模型组件已按顺序停止");
}

async fn shutdown_component(sc: &SupervisorConfig, c: &Component, timeout_secs: u64) {
    let pids = listener_pids(c.port);
    if pids.is_empty() {
        if port_open(c.port) {
            log(
                sc,
                "ERROR",
                format!("关闭 {}", c.label),
                format!("端口 :{} 有监听但找不到 pid", c.port),
            );
        } else {
            log(
                sc,
                "INFO",
                format!("关闭 {}", c.label),
                format!("未运行 :{}", c.port),
            );
        }
        return;
    }

    for pid in &pids {
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .status();
    }
    log(
        sc,
        "INFO",
        format!("关闭 {}", c.label),
        format!("SIGTERM pid={pids:?}"),
    );

    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(timeout_secs) {
        if !port_open(c.port) {
            log(
                sc,
                "INFO",
                format!("等待 {}", c.label),
                format!("已退出 {:.1}s", started.elapsed().as_secs_f64()),
            );
            return;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    log(
        sc,
        "WARN",
        format!("关闭 {}", c.label),
        format!("{timeout_secs}s 未退出，SIGKILL"),
    );
    for pid in &pids {
        let _ = Command::new("kill")
            .arg("-KILL")
            .arg(pid.to_string())
            .status();
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
    if port_open(c.port) {
        log(
            sc,
            "ERROR",
            format!("关闭 {}", c.label),
            format!("SIGKILL 后端口 :{} 仍占用", c.port),
        );
    }
}

fn listener_pids(port: u16) -> Vec<i32> {
    let output = Command::new("ss")
        .args(["-ltnp", &format!("sport = :{port}")])
        .output();
    let Ok(out) = output else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut pids = Vec::new();
    let mut rest = text.as_ref();
    while let Some(i) = rest.find("pid=") {
        rest = &rest[i + 4..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(pid) = digits.parse::<i32>()
            && !pids.contains(&pid)
        {
            pids.push(pid);
        }
    }
    pids
}

pub async fn watch(sc: &SupervisorConfig, interval_secs: u64) {
    let interval = interval_secs.max(5);
    let mut first = true;
    loop {
        ensure_all(sc, first).await;
        first = false;
        tokio::time::sleep(Duration::from_secs(interval)).await;
    }
}

async fn ensure_postgres(sc: &SupervisorConfig, verbose: bool) -> bool {
    if pg_ok(&sc.database_url).await {
        if verbose {
            log(sc, "INFO", "检查 PostgreSQL", "连接正常");
        }
        return true;
    }
    log(sc, "WARN", "检查 PostgreSQL", "连接失败，尝试拉起");
    if !sc.pgdata.join("PG_VERSION").exists() {
        log(
            sc,
            "ERROR",
            "启动 PostgreSQL",
            format!("数据目录不存在: {}", sc.pgdata.display()),
        );
        return false;
    }
    let log_path = sc.log_dir.join("postgres-start.log");
    let _ = fs::create_dir_all(&sc.log_dir);
    let output = Command::new(&sc.pg_ctl)
        .arg("-D")
        .arg(&sc.pgdata)
        .arg("-l")
        .arg(&log_path)
        .args(["start", "-w", "-t", "30"])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            log(sc, "INFO", "启动 PostgreSQL", "pg_ctl 返回 0");
        }
        Ok(o) => {
            log(
                sc,
                "WARN",
                "启动 PostgreSQL",
                format!("pg_ctl 返回 {}，继续等待", o.status),
            );
        }
        Err(e) => {
            log(sc, "ERROR", "启动 PostgreSQL", format!("{e}"));
            return false;
        }
    }

    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(30) {
        if pg_ok(&sc.database_url).await {
            log(
                sc,
                "INFO",
                "等待 PostgreSQL",
                format!("连接成功 {:.1}s", started.elapsed().as_secs_f64()),
            );
            return true;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    log(sc, "ERROR", "等待 PostgreSQL", "30s 超时");
    false
}

async fn pg_ok(database_url: &str) -> bool {
    let Ok(cfg) = database_url.parse::<tokio_postgres::Config>() else {
        return false;
    };
    match cfg.connect(NoTls).await {
        Ok((client, connection)) => {
            tokio::spawn(async move {
                let _ = connection.await;
            });
            client.simple_query("select 1").await.is_ok()
        }
        Err(_) => false,
    }
}

async fn ensure_component(sc: &SupervisorConfig, c: &Component, verbose: bool) -> bool {
    match health(c.port).await {
        Health::Up => {
            if verbose {
                log(
                    sc,
                    "INFO",
                    format!("检查 {}", c.label),
                    format!("运行中 :{}", c.port),
                );
            }
            true
        }
        Health::Loading => {
            log(
                sc,
                "WARN",
                format!("检查 {}", c.label),
                format!("正在加载 :{}", c.port),
            );
            wait_health(sc, c, 180).await
        }
        Health::Busy => {
            log(
                sc,
                "ERROR",
                format!("检查 {}", c.label),
                format!("端口被其他进程占用 :{}", c.port),
            );
            false
        }
        Health::Down => {
            log(
                sc,
                "WARN",
                format!("检查 {}", c.label),
                format!("未运行 :{}", c.port),
            );
            match spawn_script(sc, c) {
                Ok(()) => {
                    log(
                        sc,
                        "INFO",
                        format!("启动 {}", c.label),
                        format!("已派生进程，等待健康 :{}", c.port),
                    );
                    wait_health(sc, c, 180).await
                }
                Err(e) => {
                    log(sc, "ERROR", format!("启动 {}", c.label), format!("{e:#}"));
                    false
                }
            }
        }
    }
}

async fn wait_health(sc: &SupervisorConfig, c: &Component, timeout_secs: u64) -> bool {
    let started = Instant::now();
    let timeout = Duration::from_secs(timeout_secs);
    while started.elapsed() < timeout {
        match health(c.port).await {
            Health::Up => {
                log(
                    sc,
                    "INFO",
                    format!("等待 {}", c.label),
                    format!("健康 {:.1}s", started.elapsed().as_secs_f64()),
                );
                return true;
            }
            Health::Busy => {
                log(
                    sc,
                    "ERROR",
                    format!("等待 {}", c.label),
                    format!("端口被占用 :{}", c.port),
                );
                return false;
            }
            _ => tokio::time::sleep(Duration::from_secs(1)).await,
        }
    }
    log(
        sc,
        "ERROR",
        format!("等待 {}", c.label),
        format!("{timeout_secs}s 超时"),
    );
    false
}

fn spawn_script(sc: &SupervisorConfig, c: &Component) -> anyhow::Result<()> {
    let script_dir = std::env::var("NOTE_SCRIPT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts"));
    let script = script_dir.join(c.script);
    fs::create_dir_all(&sc.log_dir)?;
    let out_path = sc.log_dir.join(format!("{}-server.log", c.id));
    let out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)?;
    let err = out.try_clone()?;
    Command::new("setsid")
        .arg(&script)
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .spawn()
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("启动脚本失败 {}: {e}", script.display()))
}

async fn health(port: u16) -> Health {
    let url = format!("http://127.0.0.1:{port}/health");
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(1500))
        .build()
    {
        Ok(c) => c,
        Err(_) => return Health::Down,
    };
    match client.get(&url).send().await {
        Ok(r) if r.status().is_success() => Health::Up,
        Ok(r) if r.status().as_u16() == 503 => Health::Loading,
        Ok(_) => Health::Busy,
        Err(_) => {
            // 端口有监听但 health 不通，也按占用处理，避免重复拉起。
            if port_open(port) {
                Health::Busy
            } else {
                Health::Down
            }
        }
    }
}

fn port_open(port: u16) -> bool {
    let addr: Option<SocketAddr> = format!("127.0.0.1:{port}")
        .to_socket_addrs()
        .ok()
        .and_then(|mut it| it.next());
    match addr {
        Some(a) => TcpStream::connect_timeout(&a, Duration::from_millis(300)).is_ok(),
        None => false,
    }
}
