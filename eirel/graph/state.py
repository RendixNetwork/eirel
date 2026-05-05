"""Typed graph state with explicit per-field reducers.

State is a plain ``dict[str, Any]`` at runtime; a :class:`StateSpec`
describes which keys exist, how to merge partial node outputs into the
running state, and what the initial values are.

Why explicit reducers (no ``typing.Annotated`` magic)
-----------------------------------------------------

LangGraph leans on ``Annotated[T, reducer]`` to attach reducer functions
to ``TypedDict`` fields. That works but ties the API to a specific Python
type-system trick and makes the wiring opaque to anyone reading the
schema. Eirel's graph state is more honest: a plain registry of fields,
each carrying its reducer + default. A miner can introspect, override, or
serialize a :class:`StateSpec` without poking at type metadata.

Reducer contract
----------------

A reducer is ``Callable[[existing, update], merged]``. It MUST be pure
(no side effects), MUST handle ``existing is None`` gracefully (the spec
applies defaults before any node runs, but partial updates may still
omit a key), and MUST NOT mutate ``existing`` in place — return a new
value. The runtime executor relies on this purity to make parallel
fan-out → join semantics deterministic.

Three reducers are bundled because they cover ~90% of cases:

  * :func:`add_messages` — append-style for list-shaped fields
    (chat messages, tool results, traces).
  * :func:`merge_dict` — shallow dict merge, update wins on conflict
    (user profile, scratchpad, intermediate flags).
  * :func:`replace` — overwrite with the update value; ``None`` is a
    no-op so a node that doesn't touch the field doesn't blank it.

Custom reducers are encouraged for non-trivial fields — e.g. a tracker
that dedupes citations by URL, or one that bounds a list to N items.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Reducer",
    "StateField",
    "StateSpec",
    "add_messages",
    "merge_dict",
    "replace",
]


Reducer = Callable[[Any, Any], Any]


# -- Bundled reducers ---------------------------------------------------------


def add_messages(existing: Any, update: Any) -> list[Any]:
    """Append ``update`` to ``existing``, treating either as a list-or-scalar.

    The trailing semantics match LangGraph's ``add_messages``: if ``update``
    is a list, every entry is appended in order; if it's a single item, it's
    appended as one element. ``None`` is treated as "no change".
    """
    if update is None:
        return list(existing) if existing else []
    base: list[Any] = list(existing) if existing else []
    if isinstance(update, list):
        base.extend(update)
    else:
        base.append(update)
    return base


def merge_dict(existing: Any, update: Any) -> dict[str, Any]:
    """Shallow-merge two dicts. ``update`` keys override ``existing``."""
    if update is None:
        return dict(existing) if existing else {}
    if not isinstance(update, Mapping):
        raise TypeError(
            f"merge_dict expected a mapping update, got {type(update).__name__}"
        )
    base: dict[str, Any] = dict(existing) if existing else {}
    base.update(update)
    return base


def replace(existing: Any, update: Any) -> Any:
    """Overwrite ``existing`` with ``update``. ``None`` is a no-op.

    The no-op-on-None semantic lets a node return a partial state that
    omits a key without inadvertently blanking it. To explicitly clear a
    field, pass an empty container (``[]`` / ``{}`` / ``""``) instead.
    """
    if update is None:
        return existing
    return update


# -- Schema ------------------------------------------------------------------


_MISSING = object()


@dataclass(frozen=True, slots=True)
class StateField:
    """One field on a :class:`StateSpec`.

    Either ``default`` or ``default_factory`` should be set; if both are
    omitted the field is required at :meth:`StateSpec.init` time.
    """

    reducer: Reducer
    default: Any = _MISSING
    default_factory: Callable[[], Any] | None = None

    def initial(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        if self.default is _MISSING:
            return None
        return self.default

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING or self.default_factory is not None


class StateSpec:
    """Schema declaring the shape of a graph's state.

    Usage::

        spec = StateSpec({
            "messages": StateField(reducer=add_messages, default_factory=list),
            "scratchpad": StateField(reducer=merge_dict, default_factory=dict),
            "next": StateField(reducer=replace, default=""),
        })

        state = spec.init(messages=[{"role": "user", "content": "hi"}])
        state = spec.merge(state, {"next": "planner"})
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Mapping[str, StateField]) -> None:
        if not fields:
            raise ValueError("StateSpec requires at least one field")
        for name, field in fields.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"state field name must be a non-empty string: {name!r}")
            if not isinstance(field, StateField):
                raise TypeError(
                    f"state field {name!r} must be a StateField, got {type(field).__name__}"
                )
        self._fields: dict[str, StateField] = dict(fields)

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __getitem__(self, name: str) -> StateField:
        return self._fields[name]

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self._fields)

    def init(self, **values: Any) -> dict[str, Any]:
        """Build a fresh state dict.

        Caller-supplied ``values`` win over field defaults. Extra keys raise.
        Required fields (no default) raise if missing.
        """
        unknown = set(values) - set(self._fields)
        if unknown:
            raise KeyError(f"unknown state fields: {sorted(unknown)}")
        state: dict[str, Any] = {}
        for name, field in self._fields.items():
            if name in values:
                state[name] = values[name]
            elif field.has_default:
                state[name] = field.initial()
            else:
                raise KeyError(f"state field {name!r} is required (no default)")
        return state

    def merge(self, current: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
        """Apply per-field reducers from ``update`` onto ``current``.

        Returns a new dict; neither argument is mutated. Keys in ``update``
        that aren't declared on the spec raise ``KeyError`` — graph nodes
        must only emit declared fields. Reducers see ``current.get(name)``
        as the existing value (so a not-yet-touched key surfaces as
        ``None`` to the reducer).
        """
        unknown = set(update) - set(self._fields)
        if unknown:
            raise KeyError(f"node emitted unknown state fields: {sorted(unknown)}")
        merged: dict[str, Any] = dict(current)
        for name, value in update.items():
            field = self._fields[name]
            merged[name] = field.reducer(merged.get(name), value)
        return merged
