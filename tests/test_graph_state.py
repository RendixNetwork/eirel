from __future__ import annotations

import pytest

from eirel.graph.state import (
    StateField,
    StateSpec,
    add_messages,
    merge_dict,
    replace,
)


def test_add_messages_appends_list_or_scalar():
    assert add_messages(None, None) == []
    assert add_messages(None, {"role": "user", "content": "hi"}) == [
        {"role": "user", "content": "hi"}
    ]
    assert add_messages([{"role": "system"}], {"role": "user"}) == [
        {"role": "system"},
        {"role": "user"},
    ]
    assert add_messages([{"a": 1}], [{"b": 2}, {"c": 3}]) == [
        {"a": 1},
        {"b": 2},
        {"c": 3},
    ]


def test_add_messages_does_not_mutate_existing():
    base = [{"a": 1}]
    add_messages(base, [{"b": 2}])
    assert base == [{"a": 1}]


def test_merge_dict_shallow_overrides():
    assert merge_dict(None, None) == {}
    assert merge_dict({"a": 1}, None) == {"a": 1}
    assert merge_dict({"a": 1, "b": 2}, {"b": 99, "c": 3}) == {"a": 1, "b": 99, "c": 3}


def test_merge_dict_rejects_non_mapping_update():
    with pytest.raises(TypeError):
        merge_dict({}, ["not", "a", "dict"])


def test_replace_none_is_noop():
    assert replace("existing", None) == "existing"
    assert replace(None, "new") == "new"
    # Empty containers explicitly clear:
    assert replace([1, 2, 3], []) == []
    assert replace({"a": 1}, {}) == {}
    assert replace("x", "") == ""


def test_state_spec_init_applies_defaults():
    spec = StateSpec({
        "msgs": StateField(reducer=add_messages, default_factory=list),
        "next": StateField(reducer=replace, default="planner"),
    })
    state = spec.init()
    assert state == {"msgs": [], "next": "planner"}


def test_state_spec_init_overrides_with_kwargs():
    spec = StateSpec({
        "msgs": StateField(reducer=add_messages, default_factory=list),
    })
    state = spec.init(msgs=[{"role": "user"}])
    assert state == {"msgs": [{"role": "user"}]}


def test_state_spec_init_rejects_unknown_keys():
    spec = StateSpec({
        "msgs": StateField(reducer=add_messages, default_factory=list),
    })
    with pytest.raises(KeyError, match="unknown"):
        spec.init(other=1)


def test_state_spec_init_requires_field_without_default():
    spec = StateSpec({
        "required": StateField(reducer=replace),
    })
    with pytest.raises(KeyError, match="required"):
        spec.init()


def test_state_spec_merge_applies_per_field_reducers():
    spec = StateSpec({
        "msgs": StateField(reducer=add_messages, default_factory=list),
        "scratch": StateField(reducer=merge_dict, default_factory=dict),
        "next": StateField(reducer=replace, default=""),
    })
    state = spec.init()
    state = spec.merge(state, {"msgs": {"role": "user", "content": "hi"}})
    state = spec.merge(state, {"scratch": {"k": "v"}, "next": "planner"})
    assert state["msgs"] == [{"role": "user", "content": "hi"}]
    assert state["scratch"] == {"k": "v"}
    assert state["next"] == "planner"


def test_state_spec_merge_rejects_undeclared_keys():
    spec = StateSpec({
        "msgs": StateField(reducer=add_messages, default_factory=list),
    })
    state = spec.init()
    with pytest.raises(KeyError, match="unknown"):
        spec.merge(state, {"undeclared": 1})


def test_state_spec_does_not_mutate_input_state():
    spec = StateSpec({
        "msgs": StateField(reducer=add_messages, default_factory=list),
    })
    state = spec.init()
    merged = spec.merge(state, {"msgs": {"role": "user"}})
    assert state == {"msgs": []}
    assert merged == {"msgs": [{"role": "user"}]}


def test_state_spec_rejects_empty_or_invalid_fields():
    with pytest.raises(ValueError):
        StateSpec({})
    with pytest.raises(TypeError):
        StateSpec({"x": "not a StateField"})  # type: ignore[arg-type]
