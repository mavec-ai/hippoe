use bincode;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::memory::Trace;
use crate::types::{Emotion, Id};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRow {
    pub id: String,
    pub embedding: Vec<u8>,
    pub accesses: Vec<u8>,
    pub emotion_valence: f64,
    pub emotion_arousal: f64,
    pub wm_accessed_at: Option<i64>,
    pub outgoing: Vec<u8>,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceData {
    pub id: String,
    pub embedding: Vec<f64>,
    pub accesses: Vec<u64>,
    pub emotion_valence: f64,
    pub emotion_arousal: f64,
    pub wm_accessed_at: Option<u64>,
    pub outgoing: Vec<(String, f64)>,
    pub context: Option<String>,
}

pub fn trace_to_row(trace: &Trace) -> Result<TraceRow> {
    let embedding =
        bincode::serialize(&trace.embedding).map_err(|e| Error::Serialization(e.to_string()))?;
    let accesses =
        bincode::serialize(&trace.accesses).map_err(|e| Error::Serialization(e.to_string()))?;
    let outgoing =
        bincode::serialize(&trace.outgoing).map_err(|e| Error::Serialization(e.to_string()))?;

    Ok(TraceRow {
        id: trace.id.to_string(),
        embedding,
        accesses,
        emotion_valence: trace.emotion.valence,
        emotion_arousal: trace.emotion.arousal,
        wm_accessed_at: trace.wm_accessed_at.map(|t| t as i64),
        outgoing,
        context: trace.context.clone(),
    })
}

pub fn row_to_trace(row: TraceRow) -> Result<Trace> {
    let embedding: Vec<f64> =
        bincode::deserialize(&row.embedding).map_err(|e| Error::Serialization(e.to_string()))?;
    let accesses: Vec<u64> =
        bincode::deserialize(&row.accesses).map_err(|e| Error::Serialization(e.to_string()))?;
    let outgoing: std::collections::HashMap<Id, f64> =
        bincode::deserialize(&row.outgoing).map_err(|e| Error::Serialization(e.to_string()))?;

    let id: Id = row
        .id
        .parse()
        .map_err(|e| Error::Storage(format!("Invalid ID: {}", e)))?;

    Ok(Trace {
        id,
        embedding,
        accesses,
        emotion: Emotion {
            valence: row.emotion_valence,
            arousal: row.emotion_arousal,
        },
        wm_accessed_at: row.wm_accessed_at.map(|t| t as u64),
        outgoing,
        context: row.context,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn tuple_to_trace(
    id: String,
    embedding: Vec<u8>,
    accesses: Vec<u8>,
    emotion_valence: f64,
    emotion_arousal: f64,
    wm_accessed_at: Option<i64>,
    outgoing: Vec<u8>,
    context: Option<String>,
) -> Result<Trace> {
    row_to_trace(TraceRow {
        id,
        embedding,
        accesses,
        emotion_valence,
        emotion_arousal,
        wm_accessed_at,
        outgoing,
        context,
    })
}

pub fn trace_to_data(trace: &Trace) -> TraceData {
    let outgoing: Vec<(String, f64)> = trace
        .outgoing
        .iter()
        .map(|(id, strength)| (id.to_string(), *strength))
        .collect();

    TraceData {
        id: trace.id.to_string(),
        embedding: trace.embedding.clone(),
        accesses: trace.accesses.clone(),
        emotion_valence: trace.emotion.valence,
        emotion_arousal: trace.emotion.arousal,
        wm_accessed_at: trace.wm_accessed_at,
        outgoing,
        context: trace.context.clone(),
    }
}

pub fn data_to_trace(data: TraceData) -> Result<Trace> {
    let id: Id = data
        .id
        .parse()
        .map_err(|e| Error::Storage(format!("Invalid ID: {}", e)))?;

    let outgoing: std::collections::HashMap<Id, f64> = data
        .outgoing
        .into_iter()
        .filter_map(|(id_str, strength)| id_str.parse::<Id>().ok().map(|id| (id, strength)))
        .collect();

    Ok(Trace {
        id,
        embedding: data.embedding,
        accesses: data.accesses,
        emotion: Emotion {
            valence: data.emotion_valence,
            arousal: data.emotion_arousal,
        },
        wm_accessed_at: data.wm_accessed_at,
        outgoing,
        context: data.context,
    })
}
