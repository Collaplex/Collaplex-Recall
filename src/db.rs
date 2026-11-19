use anyhow::{Context, Result};
use tokio_postgres::{Client, NoTls};

/// 派生数据：整库删掉、从笔记原文重新 ingest 即可重建。
const SCHEMA: &str = r#"
create extension if not exists vector;

create table if not exists notes (
    id           bigserial primary key,
    path         text not null unique,
    title        text not null default '',
    content_hash text not null,
    commit_hash  text,
    updated_at   timestamptz not null default now()
);

create table if not exists chunks (
    id          bigserial primary key,
    note_id     bigint not null references notes(id) on delete cascade,
    ord         int not null,
    heading     text not null default '',
    text        text not null,
    chunk_hash  text not null,
    token_count int not null default 0,
    unique (note_id, ord)
);

-- BM25 的倒排表：term → chunk 的词频。
-- 中文按单字 + 双字建词项，不依赖 jieba/zhparser 这类扩展。
create table if not exists chunk_terms (
    chunk_id bigint not null references chunks(id) on delete cascade,
    term     text not null,
    tf       int not null,
    primary key (chunk_id, term)
);

create index if not exists chunk_terms_term_idx on chunk_terms (term);

-- 记忆库的第一张表：用户说法 → 实体。alias 唯一，冲突时由程序拒绝。
create table if not exists aliases (
    id         bigserial primary key,
    alias      text not null,
    normalized text not null unique,
    entity     text not null,
    source     text not null default 'user',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists aliases_entity_idx on aliases (entity);

-- 记忆条：事实 + provenance + 软删除。来源变了由程序降级为 stale。
create table if not exists memories (
    id                bigserial primary key,
    text              text not null,
    text_hash         text not null,
    note_id           bigint references notes(id) on delete set null,
    note_path         text,
    commit_hash       text,
    note_content_hash text,
    chunk_id          bigint,
    chunk_hash        text,
    pinned            boolean not null default false,
    status            text not null default 'active', -- active / stale / deleted
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now(),
    deleted_at        timestamptz
);

alter table memories add column if not exists chunk_id bigint;
alter table memories add column if not exists chunk_hash text;

create index if not exists memories_status_idx on memories (status);
create index if not exists memories_chunk_idx on memories (chunk_id);
create unique index if not exists memories_dedup_idx
    on memories (text_hash, coalesce(note_id, 0)) where status <> 'deleted';

-- 经验：流程规则，先进候选区，复用验证后才晋升。
create table if not exists experiences (
    id             bigserial primary key,
    text           text not null,
    text_hash      text not null,
    status         text not null default 'candidate', -- candidate / active / rejected
    uses           int not null default 0,
    source_session text,
    source_turn    int,
    created_at     timestamptz not null default now(),
    promoted_at    timestamptz
);

create unique index if not exists experiences_hash_idx
    on experiences (text_hash) where status <> 'rejected';

-- 记忆图边：只作用于记忆库，权重带衰减，纠正/失效时 LTD。
create table if not exists memory_edges (
    a_id      bigint not null references memories(id) on delete cascade,
    b_id      bigint not null references memories(id) on delete cascade,
    weight    double precision not null default 0,
    t_updated timestamptz not null default now(),
    primary key (a_id, b_id)
);

-- 补丁提案：模型只产出 new_content/diff，用户 apply 才写文件。
create table if not exists proposals (
    id          bigserial primary key,
    note_path   text not null,
    base_hash   text not null,
    old_content text not null,
    new_content text not null,
    diff        text not null,
    status      text not null default 'pending', -- pending / applied / rejected
    session_id  text,
    turn        int,
    created_at  timestamptz not null default now(),
    applied_at  timestamptz
);

create index if not exists proposals_status_idx on proposals (status);

-- 运行参数（例如 tune --apply 校准出的 α）。
create table if not exists settings (
    key        text primary key,
    value      text not null,
    updated_at timestamptz not null default now()
);

-- 反馈埋点：采纳 / 纠正 / 点引用 / 删记忆。后面调打分和边权都吃这份数据。
create table if not exists feedback (
    id         bigserial primary key,
    session_id text,
    turn       int,
    kind       text not null,
    target     text,
    detail     jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists feedback_kind_idx on feedback (kind);

-- 异步提炼的提交记录：模型产物先落 delta，校验后由程序合并。
create table if not exists memory_deltas (
    id         bigserial primary key,
    session_id text,
    turn       int,
    payload    jsonb not null,
    status     text not null default 'pending', -- pending / applied / rejected
    created_at timestamptz not null default now(),
    applied_at timestamptz
);

create table if not exists embeddings (
    chunk_id bigint not null references chunks(id) on delete cascade,
    model    text not null,
    dim      int not null,
    vec      vector not null,
    primary key (chunk_id, model)
);
"#;

pub async fn connect(database_url: &str) -> Result<Client> {
    let pg: tokio_postgres::Config = database_url
        .parse()
        .with_context(|| format!("解析连接串失败: {database_url}"))?;
    let (client, connection) = pg
        .connect(NoTls)
        .await
        .context("连接 PostgreSQL 失败（实例是否在跑？socket 路径对不对？）")?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("数据库连接断开: {e}");
        }
    });
    Ok(client)
}

pub async fn init_schema(client: &Client) -> Result<()> {
    client
        .batch_execute(SCHEMA)
        .await
        .context("初始化表结构失败（vector / pg_trgm 扩展装了吗？）")?;
    Ok(())
}
