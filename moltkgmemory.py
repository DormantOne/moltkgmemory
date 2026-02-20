"""
moltkgmemory.py — Knowledge Graph Memory for AI Agents

A SQLite-backed knowledge graph where relationships ARE the knowledge.
Nodes represent entities, concepts, events, and sources. Edges capture
how they relate: co-access patterns, contradictions, support chains,
mentions, derivations, and temporal sequences.

The "dreaming agent" consolidation pass decays stale nodes, boosts
co-accessed clusters, and surfaces unresolved contradictions — without
adding new knowledge. It reorganizes what's already there.

Schema: https://github.com/DormantOne/moltkgmemory
Built collaboratively in AICQ Journal Club, February 2026.

Zero external dependencies — just Python 3.8+ and sqlite3.
"""

import sqlite3
import uuid
import json
import math
from datetime import datetime, timezone
from typing import Optional


# --- Constants ---

NODE_TYPES = ("entity", "concept", "event", "source")

EDGE_TYPES = (
    "co_accessed",
    "contradicts",
    "supports",
    "mentions",
    "derived_from",
    "temporal_sequence",
)

RESOLUTION_STATUSES = (
    "unreviewed",
    "investigated_consistent",
    "investigated_both_valid",
    "resolved",
)

# Dreaming agent defaults
DEFAULT_DECAY_RATE = 0.05       # HP lost per day of inactivity
DEFAULT_BOOST_FACTOR = 0.1      # HP gained from co-access with hot nodes
DEFAULT_STALE_DAYS = 7          # Days without access before a node is "stale"
DEFAULT_MIN_CONFIDENCE = 0.01   # Floor — nodes never fully vanish


def _now() -> str:
    """ISO 8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    """Fresh UUID string."""
    return str(uuid.uuid4())


class MoltKGMemory:
    """
    Knowledge graph memory backed by SQLite.

    Usage:
        kg = MoltKGMemory("my_memory.db")
        nid = kg.add_node("entity", "AlanBotts", "Curator of StrangerLoops")
        kg.add_edge(nid, other_id, "supports", weight=0.8)
        kg.touch(nid)
        results = kg.search("StrangerLoops")
        report = kg.dream()
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    # --- Schema Setup ---

    def _create_tables(self):
        c = self.conn
        c.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                label TEXT NOT NULL,
                content TEXT DEFAULT '',
                confidence REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                source_ids TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                type TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                last_reinforced TEXT NOT NULL,
                context_ids TEXT DEFAULT '[]',
                resolution_status TEXT DEFAULT NULL,
                FOREIGN KEY (source) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target) REFERENCES nodes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
            CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
            CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON nodes(confidence);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type);
        """)
        c.commit()

    # --- Node Operations ---

    def add_node(
        self,
        node_type: str,
        label: str,
        content: str = "",
        confidence: float = 1.0,
        tags: Optional[list] = None,
        source_ids: Optional[list] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """
        Add a node to the knowledge graph.

        Args:
            node_type: One of entity, concept, event, source.
            label: Human-readable name for the node.
            content: Descriptive text about this node.
            confidence: Initial confidence score (0.0 to 1.0).
            tags: Optional list of tag strings.
            source_ids: Optional list of source node UUIDs.
            node_id: Optional explicit UUID (auto-generated if omitted).

        Returns:
            The UUID of the created node.
        """
        if node_type not in NODE_TYPES:
            raise ValueError(f"node_type must be one of {NODE_TYPES}, got '{node_type}'")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")

        nid = node_id or _uuid()
        now = _now()
        self.conn.execute(
            """INSERT INTO nodes (id, type, label, content, confidence,
               access_count, created_at, last_accessed, tags, source_ids)
               VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)""",
            (
                nid,
                node_type,
                label,
                content,
                confidence,
                now,
                now,
                json.dumps(tags or []),
                json.dumps(source_ids or []),
            ),
        )
        self.conn.commit()
        return nid

    def get_node(self, node_id: str) -> Optional[dict]:
        """Retrieve a single node by ID. Returns None if not found."""
        row = self.conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def touch(self, node_id: str) -> dict:
        """
        Record an access to this node. Increments access_count,
        updates last_accessed, and gives a small confidence boost
        (capped at 1.0).

        Returns the updated node dict.
        """
        now = _now()
        self.conn.execute(
            """UPDATE nodes
               SET access_count = access_count + 1,
                   last_accessed = ?,
                   confidence = MIN(1.0, confidence + 0.02)
               WHERE id = ?""",
            (now, node_id),
        )
        self.conn.commit()
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(f"Node not found: {node_id}")
        return node

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its edges (via CASCADE). Returns True if deleted."""
        cur = self.conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # --- Edge Operations ---

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        weight: float = 0.5,
        context_ids: Optional[list] = None,
        resolution_status: Optional[str] = None,
        edge_id: Optional[str] = None,
    ) -> str:
        """
        Add an edge between two nodes.

        Args:
            source: UUID of the source node.
            target: UUID of the target node.
            edge_type: One of co_accessed, contradicts, supports, mentions,
                       derived_from, temporal_sequence.
            weight: Strength of the relationship (0.0 to 1.0).
            context_ids: Optional list of context UUIDs.
            resolution_status: For contradicts edges — one of unreviewed,
                               investigated_consistent, investigated_both_valid,
                               resolved.
            edge_id: Optional explicit UUID.

        Returns:
            The UUID of the created edge.
        """
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"edge_type must be one of {EDGE_TYPES}, got '{edge_type}'")
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be between 0.0 and 1.0, got {weight}")
        if edge_type == "contradicts" and resolution_status is None:
            resolution_status = "unreviewed"
        if resolution_status is not None and resolution_status not in RESOLUTION_STATUSES:
            raise ValueError(
                f"resolution_status must be one of {RESOLUTION_STATUSES}, "
                f"got '{resolution_status}'"
            )

        eid = edge_id or _uuid()
        now = _now()
        self.conn.execute(
            """INSERT INTO edges (id, source, target, type, weight,
               created_at, last_reinforced, context_ids, resolution_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                eid,
                source,
                target,
                edge_type,
                weight,
                now,
                now,
                json.dumps(context_ids or []),
                resolution_status,
            ),
        )
        self.conn.commit()
        return eid

    def get_edge(self, edge_id: str) -> Optional[dict]:
        """Retrieve a single edge by ID."""
        row = self.conn.execute(
            "SELECT * FROM edges WHERE id = ?", (edge_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_edge(row)

    def reinforce_edge(self, edge_id: str, boost: float = 0.1) -> dict:
        """
        Reinforce an existing edge — increases weight (capped at 1.0)
        and updates last_reinforced timestamp.
        """
        now = _now()
        self.conn.execute(
            """UPDATE edges
               SET weight = MIN(1.0, weight + ?),
                   last_reinforced = ?
               WHERE id = ?""",
            (boost, now, edge_id),
        )
        self.conn.commit()
        edge = self.get_edge(edge_id)
        if edge is None:
            raise KeyError(f"Edge not found: {edge_id}")
        return edge

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge. Returns True if deleted."""
        cur = self.conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # --- Query Operations ---

    def neighbors(self, node_id: str, edge_type: Optional[str] = None) -> list:
        """
        Get all nodes connected to the given node (in either direction).
        Optionally filter by edge type.

        Returns list of dicts: {"node": {...}, "edge": {...}, "direction": "outgoing"|"incoming"}
        """
        results = []

        # Outgoing edges (this node is source)
        query = "SELECT * FROM edges WHERE source = ?"
        params = [node_id]
        if edge_type:
            query += " AND type = ?"
            params.append(edge_type)

        for row in self.conn.execute(query, params).fetchall():
            edge = self._row_to_edge(row)
            target_node = self.get_node(edge["target"])
            if target_node:
                results.append({"node": target_node, "edge": edge, "direction": "outgoing"})

        # Incoming edges (this node is target)
        query = "SELECT * FROM edges WHERE target = ?"
        params = [node_id]
        if edge_type:
            query += " AND type = ?"
            params.append(edge_type)

        for row in self.conn.execute(query, params).fetchall():
            edge = self._row_to_edge(row)
            source_node = self.get_node(edge["source"])
            if source_node:
                results.append({"node": source_node, "edge": edge, "direction": "incoming"})

        return results

    def search(self, query: str, node_type: Optional[str] = None, limit: int = 20) -> list:
        """
        Search nodes by label or content (case-insensitive substring match).
        Optionally filter by node type. Results sorted by confidence descending.
        """
        sql = "SELECT * FROM nodes WHERE (label LIKE ? OR content LIKE ?)"
        params = [f"%{query}%", f"%{query}%"]

        if node_type:
            if node_type not in NODE_TYPES:
                raise ValueError(f"node_type must be one of {NODE_TYPES}")
            sql += " AND type = ?"
            params.append(node_type)

        sql += " ORDER BY confidence DESC, access_count DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_contradictions(self, status: str = "unreviewed") -> list:
        """
        Get all contradiction edges with the given resolution status.
        Returns list of dicts with source_node, target_node, and edge info.
        """
        if status not in RESOLUTION_STATUSES:
            raise ValueError(f"status must be one of {RESOLUTION_STATUSES}")

        rows = self.conn.execute(
            """SELECT * FROM edges
               WHERE type = 'contradicts' AND resolution_status = ?
               ORDER BY weight DESC""",
            (status,),
        ).fetchall()

        results = []
        for row in rows:
            edge = self._row_to_edge(row)
            src = self.get_node(edge["source"])
            tgt = self.get_node(edge["target"])
            if src and tgt:
                results.append({
                    "source_node": src,
                    "target_node": tgt,
                    "edge": edge,
                })
        return results

    # --- The Dreaming Agent ---

    def dream(
        self,
        decay_rate: float = DEFAULT_DECAY_RATE,
        boost_factor: float = DEFAULT_BOOST_FACTOR,
        stale_days: float = DEFAULT_STALE_DAYS,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> dict:
        """
        Run the dreaming agent consolidation pass.

        This does NOT add new knowledge. It reorganizes existing knowledge:

        1. DECAY: Nodes not accessed in stale_days lose confidence.
           Loss scales with days of inactivity.
        2. BOOST: Nodes co-accessed with high-confidence nodes get a lift.
           Knowledge clusters strengthen together.
        3. SURFACE: Unreviewed contradictions are collected for attention.
        4. REPORT: Returns a structured summary of what changed.

        Args:
            decay_rate: Confidence lost per day of inactivity (default 0.05).
            boost_factor: Confidence gained from co-access with hot nodes (default 0.1).
            stale_days: Days without access before decay starts (default 7).
            min_confidence: Floor confidence — nodes never fully vanish (default 0.01).

        Returns:
            Dict with keys: decayed, boosted, contradictions, stats.
        """
        now = datetime.now(timezone.utc)
        decayed_nodes = []
        boosted_nodes = []

        # --- Phase 1: Decay stale nodes ---
        all_nodes = self.conn.execute("SELECT * FROM nodes").fetchall()

        for row in all_nodes:
            last_accessed = datetime.fromisoformat(row["last_accessed"])
            days_idle = (now - last_accessed).total_seconds() / 86400.0

            if days_idle > stale_days:
                # Exponential decay — sharper drop for very old nodes
                decay = decay_rate * math.log1p(days_idle - stale_days)
                new_confidence = max(min_confidence, row["confidence"] - decay)

                if new_confidence < row["confidence"]:
                    self.conn.execute(
                        "UPDATE nodes SET confidence = ? WHERE id = ?",
                        (new_confidence, row["id"]),
                    )
                    decayed_nodes.append({
                        "id": row["id"],
                        "label": row["label"],
                        "old_confidence": round(row["confidence"], 4),
                        "new_confidence": round(new_confidence, 4),
                        "days_idle": round(days_idle, 1),
                    })

        # --- Phase 2: Boost co-accessed clusters ---
        # Find "hot" nodes: recently accessed, high confidence
        hot_threshold = 0.7
        hot_nodes = self.conn.execute(
            "SELECT id FROM nodes WHERE confidence >= ?", (hot_threshold,)
        ).fetchall()
        hot_ids = {r["id"] for r in hot_nodes}

        for hot_id in hot_ids:
            # Find nodes co-accessed with this hot node
            co_edges = self.conn.execute(
                """SELECT * FROM edges
                   WHERE type = 'co_accessed'
                     AND (source = ? OR target = ?)""",
                (hot_id, hot_id),
            ).fetchall()

            for edge in co_edges:
                # The neighbor is whichever end isn't the hot node
                neighbor_id = edge["target"] if edge["source"] == hot_id else edge["source"]

                # Don't boost already-hot nodes
                if neighbor_id in hot_ids:
                    continue

                neighbor = self.conn.execute(
                    "SELECT * FROM nodes WHERE id = ?", (neighbor_id,)
                ).fetchone()
                if neighbor is None:
                    continue

                # Boost proportional to edge weight
                boost = boost_factor * edge["weight"]
                new_confidence = min(1.0, neighbor["confidence"] + boost)

                if new_confidence > neighbor["confidence"]:
                    self.conn.execute(
                        "UPDATE nodes SET confidence = ? WHERE id = ?",
                        (new_confidence, neighbor_id),
                    )
                    boosted_nodes.append({
                        "id": neighbor_id,
                        "label": neighbor["label"],
                        "old_confidence": round(neighbor["confidence"], 4),
                        "new_confidence": round(new_confidence, 4),
                        "boosted_by": hot_id,
                        "edge_weight": edge["weight"],
                    })

        self.conn.commit()

        # --- Phase 3: Surface contradictions ---
        contradictions = self.get_contradictions("unreviewed")

        # --- Phase 4: Build report ---
        return {
            "timestamp": _now(),
            "decayed": decayed_nodes,
            "boosted": boosted_nodes,
            "contradictions": [
                {
                    "source": c["source_node"]["label"],
                    "target": c["target_node"]["label"],
                    "weight": c["edge"]["weight"],
                    "edge_id": c["edge"]["id"],
                }
                for c in contradictions
            ],
            "stats": self.stats(),
        }

    # --- Stats ---

    def stats(self) -> dict:
        """Aggregate statistics about the knowledge graph."""
        node_count = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

        type_counts = {}
        for row in self.conn.execute(
            "SELECT type, COUNT(*) as cnt FROM nodes GROUP BY type"
        ).fetchall():
            type_counts[row["type"]] = row["cnt"]

        edge_type_counts = {}
        for row in self.conn.execute(
            "SELECT type, COUNT(*) as cnt FROM edges GROUP BY type"
        ).fetchall():
            edge_type_counts[row["type"]] = row["cnt"]

        avg_confidence = self.conn.execute(
            "SELECT AVG(confidence) FROM nodes"
        ).fetchone()[0]

        unreviewed = self.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE type='contradicts' AND resolution_status='unreviewed'"
        ).fetchone()[0]

        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "node_types": type_counts,
            "edge_types": edge_type_counts,
            "avg_confidence": round(avg_confidence, 4) if avg_confidence else 0.0,
            "unreviewed_contradictions": unreviewed,
        }

    # --- Internal Helpers ---

    def _row_to_node(self, row: sqlite3.Row) -> dict:
        """Convert a SQLite Row to a node dict matching the schema."""
        return {
            "id": row["id"],
            "type": row["type"],
            "label": row["label"],
            "content": row["content"],
            "metadata": {
                "created_at": row["created_at"],
                "last_accessed": row["last_accessed"],
                "access_count": row["access_count"],
                "source_ids": json.loads(row["source_ids"]),
                "confidence": row["confidence"],
                "tags": json.loads(row["tags"]),
            },
        }

    def _row_to_edge(self, row: sqlite3.Row) -> dict:
        """Convert a SQLite Row to an edge dict matching the schema."""
        d = {
            "id": row["id"],
            "source": row["source"],
            "target": row["target"],
            "type": row["type"],
            "weight": row["weight"],
            "metadata": {
                "created_at": row["created_at"],
                "last_reinforced": row["last_reinforced"],
                "context_ids": json.loads(row["context_ids"]),
            },
        }
        if row["resolution_status"] is not None:
            d["metadata"]["resolution_status"] = row["resolution_status"]
        return d

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        s = self.stats()
        return (
            f"MoltKGMemory({self.db_path!r}, "
            f"nodes={s['total_nodes']}, edges={s['total_edges']})"
        )
