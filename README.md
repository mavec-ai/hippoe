# hippoe

Cognitive memory system with scientifically-grounded retrieval algorithms.

## Overview

hippoe implements a memory system inspired by cognitive psychology research. It provides intelligent storage and retrieval using multiple cognitive theories working together.

## Key Features

- **ACT-R Base-Level Activation**: Memory strength decays over time but increases with use
- **MINERVA 2 Similarity**: Cubed cosine similarity amplifies strong matches
- **Spreading Activation**: Related memories boost each other through association networks
- **Temporal Context Model**: Memories linked by temporal proximity
- **Emotional Modulation**: Emotional significance affects memory strength (Circumplex Model)
- **Reconsolidation**: Reactivated memories become labile and can be strengthened
- **Session-based Decay**: Memory strength decays progressively over time

## Hybrid Retrieval Strategy

Default weights optimized for cognitive quality:

| Factor | Weight | Description |
|--------|--------|-------------|
| Similarity | 1.0 | MINERVA 2 cubed cosine similarity |
| Base-level | 0.8 | ACT-R activation based on access patterns |
| Spreading | 0.7 | Activation from related memories |
| Emotional | 0.5 | Emotional significance boost |
| Contextual | 0.3 | Context similarity |
| Temporal | 0.3 | Temporal context model links |

## Thread Safety

All public methods use immutable `&self` for concurrent access. Interior mutability via `RwLock` ensures thread-safe operations without external synchronization.

## Performance

- Batch similarity: Pre-computed probe norms for efficient retrieval
- Simplified activation: ln(n+1) - d·ln(t) approximation for base-level computation
- Concurrent access: Multiple readers, single writer pattern

## References

This implementation is grounded in peer-reviewed cognitive psychology research:

- Anderson, J. R. (1996). ACT: A simple theory of complex cognition. *American Psychologist*, 51(4), 355-365. DOI:10.1037/0003-066X.51.4.355

- Hintzman, D. L. (1986). "Schema abstraction" in a multiple-trace memory model. *Psychological Review*, 93(4), 411-428. DOI:10.1037/0033-295X.93.4.411

- Howard, M. W., & Kahana, M. J. (2002). A distributed representation of temporal context. *Journal of Mathematical Psychology*, 46(3), 269-299. DOI:10.1006/jmps.2001.1388

- Nader, K., Schafe, G. E., & Le Doux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature*, 406(6797), 722-726. DOI:10.1038/35021052

- Anderson, J. R. (1983). A spreading activation theory of memory. *Journal of Verbal Learning and Verbal Behavior*, 22(3), 261-295. DOI:10.1016/S0022-5371(83)90201-3

- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178. DOI:10.1037/h0077714
