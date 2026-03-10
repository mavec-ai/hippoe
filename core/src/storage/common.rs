use bincode;
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "postgres", feature = "sqlite"))]
use sqlx::FromRow;

use crate::error::{Error, Result};
use crate::memory::{Association, Memory, MemoryContent, MemoryMetadata};
use crate::types::{Emotion, Id, LinkKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(any(feature = "postgres", feature = "sqlite"), derive(FromRow))]
pub struct MemoryRow {
    pub id: String,
    pub content_text: Option<String>,
    pub content_structured: Option<String>,
    pub content_raw: Option<Vec<u8>>,
    pub embedding: Vec<u8>,
    pub created_at: u64,
    pub updated_at: u64,
    pub last_accessed_at: u64,
    pub access_count: u64,
    pub access_timestamps: Vec<u8>,
    pub emotion_valence: f64,
    pub emotion_arousal: f64,
    pub decay_rate: f64,
    pub importance: f64,
    pub context: Option<String>,
    pub tags: Vec<u8>,
    pub lability: f64,
    pub consolidation_threshold: f64,
    pub associations: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryData {
    pub id: String,
    pub content_text: Option<String>,
    pub content_structured: Option<String>,
    pub content_raw: Option<Vec<u8>>,
    pub embedding: Vec<f64>,
    pub created_at: u64,
    pub updated_at: u64,
    pub last_accessed_at: u64,
    pub access_count: u64,
    pub access_timestamps: Vec<u64>,
    pub emotion_valence: f64,
    pub emotion_arousal: f64,
    pub decay_rate: f64,
    pub importance: f64,
    pub context: Option<String>,
    pub tags: Vec<String>,
    pub lability: f64,
    pub consolidation_threshold: f64,
    pub associations: Vec<AssociationData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationData {
    pub target_id: String,
    pub strength: f64,
    pub kind: String,
    pub created_at: u64,
    pub last_activated: u64,
    pub activation_count: u64,
}

fn link_kind_to_string(kind: LinkKind) -> String {
    match kind {
        LinkKind::Semantic => "semantic".to_string(),
        LinkKind::Episodic => "episodic".to_string(),
        LinkKind::Temporal => "temporal".to_string(),
        LinkKind::Causal => "causal".to_string(),
    }
}

fn string_to_link_kind(s: &str) -> LinkKind {
    match s {
        "semantic" => LinkKind::Semantic,
        "episodic" => LinkKind::Episodic,
        "temporal" => LinkKind::Temporal,
        "causal" => LinkKind::Causal,
        _ => LinkKind::Semantic,
    }
}

pub fn memory_to_row(memory: &Memory) -> Result<MemoryRow> {
    let embedding =
        bincode::serialize(&memory.embedding).map_err(|e| Error::Serialization(e.to_string()))?;
    let access_timestamps = bincode::serialize(&memory.metadata.access_timestamps)
        .map_err(|e| Error::Serialization(e.to_string()))?;
    let tags = bincode::serialize(&memory.metadata.tags)
        .map_err(|e| Error::Serialization(e.to_string()))?;
    let associations = bincode::serialize(
        &memory
            .associations
            .iter()
            .map(|a| AssociationData {
                target_id: a.target_id.to_string(),
                strength: a.strength,
                kind: link_kind_to_string(a.kind),
                created_at: a.created_at,
                last_activated: a.last_activated,
                activation_count: a.activation_count,
            })
            .collect::<Vec<_>>(),
    )
    .map_err(|e| Error::Serialization(e.to_string()))?;

    Ok(MemoryRow {
        id: memory.id.to_string(),
        content_text: memory.content.text.clone(),
        content_structured: memory.content.structured.as_ref().map(|v| v.to_string()),
        content_raw: memory.content.raw.clone(),
        embedding,
        created_at: memory.metadata.created_at,
        updated_at: memory.metadata.updated_at,
        last_accessed_at: memory.metadata.last_accessed_at,
        access_count: memory.metadata.access_count,
        access_timestamps,
        emotion_valence: memory.metadata.emotional_weight.valence,
        emotion_arousal: memory.metadata.emotional_weight.arousal,
        decay_rate: memory.metadata.decay_rate,
        importance: memory.metadata.importance,
        context: memory.metadata.context.clone(),
        tags,
        lability: memory.metadata.lability,
        consolidation_threshold: memory.metadata.consolidation_threshold,
        associations,
    })
}

pub fn row_to_memory(row: MemoryRow) -> Result<Memory> {
    let id: Id = row
        .id
        .parse()
        .map_err(|e| Error::Storage(format!("Invalid ID: {}", e)))?;

    let embedding: Vec<f64> =
        bincode::deserialize(&row.embedding).map_err(|e| Error::Serialization(e.to_string()))?;
    let access_timestamps: Vec<u64> = bincode::deserialize(&row.access_timestamps)
        .map_err(|e| Error::Serialization(e.to_string()))?;
    let tags: Vec<String> =
        bincode::deserialize(&row.tags).map_err(|e| Error::Serialization(e.to_string()))?;
    let association_data: Vec<AssociationData> =
        bincode::deserialize(&row.associations).map_err(|e| Error::Serialization(e.to_string()))?;

    let associations: Vec<Association> = association_data
        .into_iter()
        .filter_map(|a| {
            a.target_id.parse::<Id>().ok().map(|target_id| Association {
                target_id,
                strength: a.strength,
                kind: string_to_link_kind(&a.kind),
                created_at: a.created_at,
                last_activated: a.last_activated,
                activation_count: a.activation_count,
            })
        })
        .collect();

    let content = MemoryContent {
        text: row.content_text,
        structured: row.content_structured.and_then(|s| serde_json::from_str(&s).ok()),
        raw: row.content_raw,
    };

    Ok(Memory {
        id,
        content,
        embedding,
        metadata: MemoryMetadata {
            created_at: row.created_at,
            updated_at: row.updated_at,
            last_accessed_at: row.last_accessed_at,
            access_count: row.access_count,
            access_timestamps,
            emotional_weight: Emotion {
                valence: row.emotion_valence,
                arousal: row.emotion_arousal,
            },
            decay_rate: row.decay_rate,
            importance: row.importance,
            context: row.context,
            tags,
            lability: row.lability,
            consolidation_threshold: row.consolidation_threshold,
        },
        associations,
    })
}

pub fn memory_to_data(memory: &Memory) -> MemoryData {
    MemoryData {
        id: memory.id.to_string(),
        content_text: memory.content.text.clone(),
        content_structured: memory.content.structured.as_ref().map(|v| v.to_string()),
        content_raw: memory.content.raw.clone(),
        embedding: memory.embedding.clone(),
        created_at: memory.metadata.created_at,
        updated_at: memory.metadata.updated_at,
        last_accessed_at: memory.metadata.last_accessed_at,
        access_count: memory.metadata.access_count,
        access_timestamps: memory.metadata.access_timestamps.clone(),
        emotion_valence: memory.metadata.emotional_weight.valence,
        emotion_arousal: memory.metadata.emotional_weight.arousal,
        decay_rate: memory.metadata.decay_rate,
        importance: memory.metadata.importance,
        context: memory.metadata.context.clone(),
        tags: memory.metadata.tags.clone(),
        lability: memory.metadata.lability,
        consolidation_threshold: memory.metadata.consolidation_threshold,
        associations: memory
            .associations
            .iter()
            .map(|a| AssociationData {
                target_id: a.target_id.to_string(),
                strength: a.strength,
                kind: link_kind_to_string(a.kind),
                created_at: a.created_at,
                last_activated: a.last_activated,
                activation_count: a.activation_count,
            })
            .collect(),
    }
}

pub fn data_to_memory(data: MemoryData) -> Result<Memory> {
    let id: Id = data
        .id
        .parse()
        .map_err(|e| Error::Storage(format!("Invalid ID: {}", e)))?;

    let associations: Vec<Association> = data
        .associations
        .into_iter()
        .filter_map(|a| {
            a.target_id.parse::<Id>().ok().map(|target_id| Association {
                target_id,
                strength: a.strength,
                kind: string_to_link_kind(&a.kind),
                created_at: a.created_at,
                last_activated: a.last_activated,
                activation_count: a.activation_count,
            })
        })
        .collect();

    let content = MemoryContent {
        text: data.content_text,
        structured: data.content_structured.and_then(|s| serde_json::from_str(&s).ok()),
        raw: data.content_raw,
    };

    Ok(Memory {
        id,
        content,
        embedding: data.embedding,
        metadata: MemoryMetadata {
            created_at: data.created_at,
            updated_at: data.updated_at,
            last_accessed_at: data.last_accessed_at,
            access_count: data.access_count,
            access_timestamps: data.access_timestamps,
            emotional_weight: Emotion {
                valence: data.emotion_valence,
                arousal: data.emotion_arousal,
            },
            decay_rate: data.decay_rate,
            importance: data.importance,
            context: data.context,
            tags: data.tags,
            lability: data.lability,
            consolidation_threshold: data.consolidation_threshold,
        },
        associations,
    })
}
