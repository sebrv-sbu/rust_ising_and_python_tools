#!/bin/bash
# Usage: append_locked.sh <lockfile> <src> <dest>
#
# AI GENERATED CODE - Claude Sonnet 5
LOCKFILE="$1"
SRC="$2"
DEST="$3"

(
  flock -x 200
  cat "$SRC" >> "$DEST"
) 200>"$LOCKFILE"
