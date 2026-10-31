ONCALL_MD = b"""# On-call handbook

## Rotation

On-call runs in weekly shifts that start every Monday at 10:00 local time.
Each shift has a primary and a secondary engineer.

## Severity

The primary must acknowledge a SEV1 page within 15 minutes and open an incident channel.
SEV3 issues are handled during business hours the next working day.

## Compensation

Engineers receive one day of time off in lieu for each week of primary on-call.
"""

CACHING_TXT = b"""HTTP caching notes.

An ETag identifies a specific version of a resource. When the ETag still matches,
the server answers 304 Not Modified with no body, which saves bandwidth.

The Vary header lists request headers that change the response, so caches store
separate variants for each value.
"""

VECTOR_PAGES = [
    "Introduction to vector search. Embeddings map text to points in space.",
    "HNSW builds a multi-layer graph for approximate nearest neighbour search.\n"
    "IVFFlat clusters vectors into lists and should be built after loading data.",
]
