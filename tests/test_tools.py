from __future__ import annotations

import pytest

from theseus.tools import (
    BashTool,
    EditTool,
    FindTool,
    GrepTool,
    LsTool,
    ReadTool,
    WriteTool,
    all_tools,
    coding_tools,
    read_only_tools,
)


# --- read -----------------------------------------------------------------------
def test_read_returns_file_contents(tmp_path):
    (tmp_path / "hello.txt").write_text("line1\nline2\nline3\n")
    result = ReadTool(tmp_path).execute(path="hello.txt")
    assert not result.is_error
    assert "line1" in result.content and "line3" in result.content


def test_read_offset_and_limit(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\nc\nd\ne\n")
    result = ReadTool(tmp_path).execute(path="f.txt", offset=2, limit=2)
    assert result.content.splitlines() == ["b", "c"]


def test_read_missing_file_is_error(tmp_path):
    result = ReadTool(tmp_path).execute(path="nope.txt")
    assert result.is_error and "not found" in result.content.lower()


def test_read_offset_out_of_range_is_error(tmp_path):
    (tmp_path / "f.txt").write_text("only\n")
    result = ReadTool(tmp_path).execute(path="f.txt", offset=99)
    assert result.is_error and "out of range" in result.content


# --- write ----------------------------------------------------------------------
def test_write_creates_file_and_parents(tmp_path):
    result = WriteTool(tmp_path).execute(path="sub/dir/new.txt", content="hi")
    assert not result.is_error
    assert (tmp_path / "sub" / "dir" / "new.txt").read_text() == "hi"


def test_write_overwrites(tmp_path):
    (tmp_path / "f.txt").write_text("old")
    WriteTool(tmp_path).execute(path="f.txt", content="new")
    assert (tmp_path / "f.txt").read_text() == "new"


# --- edit -----------------------------------------------------------------------
def test_edit_replaces_unique_match(tmp_path):
    (tmp_path / "f.txt").write_text("foo bar baz\n")
    result = EditTool(tmp_path).execute(
        path="f.txt", edits=[{"oldText": "bar", "newText": "QUX"}]
    )
    assert not result.is_error
    assert (tmp_path / "f.txt").read_text() == "foo QUX baz\n"


def test_edit_multiple_non_overlapping(tmp_path):
    (tmp_path / "f.txt").write_text("alpha beta gamma\n")
    result = EditTool(tmp_path).execute(
        path="f.txt",
        edits=[{"oldText": "alpha", "newText": "A"}, {"oldText": "gamma", "newText": "G"}],
    )
    assert not result.is_error
    assert (tmp_path / "f.txt").read_text() == "A beta G\n"


def test_edit_non_unique_is_error(tmp_path):
    (tmp_path / "f.txt").write_text("x x x\n")
    result = EditTool(tmp_path).execute(path="f.txt", edits=[{"oldText": "x", "newText": "y"}])
    assert result.is_error and "not unique" in result.content


def test_edit_missing_text_is_error(tmp_path):
    (tmp_path / "f.txt").write_text("hello\n")
    result = EditTool(tmp_path).execute(path="f.txt", edits=[{"oldText": "zzz", "newText": "q"}])
    assert result.is_error and "not found" in result.content


def test_edit_preserves_crlf(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"one\r\ntwo\r\n")
    EditTool(tmp_path).execute(path="f.txt", edits=[{"oldText": "two", "newText": "TWO"}])
    assert (tmp_path / "f.txt").read_bytes() == b"one\r\nTWO\r\n"


# --- ls -------------------------------------------------------------------------
def test_ls_lists_sorted_with_dir_suffix(tmp_path):
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "sub").mkdir()
    result = LsTool(tmp_path).execute()
    assert result.content.splitlines() == ["a.txt", "b.txt", "sub/"]


def test_ls_not_a_directory_is_error(tmp_path):
    (tmp_path / "f.txt").write_text("")
    result = LsTool(tmp_path).execute(path="f.txt")
    assert result.is_error


# --- find -----------------------------------------------------------------------
def test_find_matches_glob_recursively(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    result = FindTool(tmp_path).execute(pattern="*.py")
    found = set(result.content.splitlines())
    assert "a.py" in found and "pkg/b.py" in found and "c.txt" not in result.content


def test_find_prunes_noise_dirs(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hook.py").write_text("")
    (tmp_path / "real.py").write_text("")
    result = FindTool(tmp_path).execute(pattern="*.py")
    assert "real.py" in result.content and ".git" not in result.content


def test_find_no_matches(tmp_path):
    result = FindTool(tmp_path).execute(pattern="*.nope")
    assert "no matches" in result.content


# --- grep -----------------------------------------------------------------------
def test_grep_finds_matches_with_line_numbers(tmp_path):
    (tmp_path / "f.txt").write_text("alpha\nneedle here\nomega\n")
    result = GrepTool(tmp_path).execute(pattern="needle", path="f.txt")
    assert not result.is_error
    assert "2" in result.content and "needle" in result.content


def test_grep_literal_and_ignorecase(tmp_path):
    (tmp_path / "f.txt").write_text("Hello World\n")
    result = GrepTool(tmp_path).execute(pattern="hello", path="f.txt", ignoreCase=True)
    assert "Hello World" in result.content


def test_grep_no_matches(tmp_path):
    (tmp_path / "f.txt").write_text("nothing\n")
    result = GrepTool(tmp_path).execute(pattern="zzz", path="f.txt")
    assert "no matches" in result.content


# --- bash -----------------------------------------------------------------------
def test_bash_runs_command(tmp_path):
    result = BashTool(tmp_path).execute(command="echo hello")
    assert not result.is_error and "hello" in result.content
    assert result.details["exit_code"] == 0


def test_bash_nonzero_exit_is_error(tmp_path):
    result = BashTool(tmp_path).execute(command="exit 3")
    assert result.is_error and result.details["exit_code"] == 3


def test_bash_runs_in_cwd(tmp_path):
    (tmp_path / "marker.txt").write_text("")
    result = BashTool(tmp_path).execute(command="ls")
    assert "marker.txt" in result.content


def test_bash_timeout_is_error(tmp_path):
    result = BashTool(tmp_path).execute(command="sleep 5", timeout=1)
    assert result.is_error and result.details["timed_out"] is True


# --- registry -------------------------------------------------------------------
def test_registry_groupings():
    assert set(all_tools()) == {"read", "write", "edit", "ls", "find", "grep", "bash"}
    assert set(read_only_tools()) == {"read", "ls", "find", "grep"}
    assert set(coding_tools()) == {"read", "write", "edit", "bash"}


def test_registry_tools_have_json_schema_params():
    for tool in all_tools().values():
        assert tool.parameters["type"] == "object"
        assert "properties" in tool.parameters
