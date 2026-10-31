# PostgreSQL indexing notes

## B-tree

B-tree is the default index type in PostgreSQL. It handles equality and range comparisons
(`=`, `<`, `>`, `BETWEEN`) and can also satisfy `ORDER BY` without a separate sort step.
A multicolumn B-tree index is most useful when queries filter on the leading column; an
index on `(tenant_id, created_at)` helps a query on `tenant_id` alone but not one on
`created_at` alone.

## GIN

GIN (Generalized Inverted Index) maps each element to the rows that contain it. It is the
usual choice for full-text search on `tsvector` columns, for `jsonb` containment queries and
for array operators. GIN indexes are slower to update than B-tree indexes, so tables with
heavy write traffic pay a cost for them. The `fastupdate` storage parameter batches pending
entries to soften that cost.

## BRIN

BRIN (Block Range Index) stores a summary, such as the minimum and maximum value, for each
range of table blocks. It is tiny compared to a B-tree and works well for very large tables
where the column correlates with physical order, like an append-only events table ordered
by timestamp.

## Partial and expression indexes

A partial index only covers rows that match a `WHERE` clause, for example
`CREATE INDEX ON orders (customer_id) WHERE status = 'open'`. It stays small and fast when
queries always include the same predicate. An expression index such as
`CREATE INDEX ON users (lower(email))` lets case-insensitive lookups use an index.

## Vector indexes with pgvector

pgvector offers two approximate nearest neighbour index types. HNSW builds a multi-layer
graph; it has better recall for the same speed and does not need training data, but it uses
more memory and builds more slowly. IVFFlat clusters vectors into lists and searches only the
closest lists; it builds faster but should be created after the table already has data,
because the cluster centres are computed at build time. Each index is created for one
distance operator, such as `vector_cosine_ops` for cosine distance.
