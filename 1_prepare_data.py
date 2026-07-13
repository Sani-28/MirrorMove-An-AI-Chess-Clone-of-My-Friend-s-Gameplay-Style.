"""
STEP 1 — DATA PREPARATION
=========================
Parses PGN files with python-chess and converts every move played by YOUR
FRIEND into an OpenAI chat fine-tuning example (JSONL).

Each training instance:
  system    -> persona profile (style-mimicking prompt)
  user      -> full move history (SAN) up to that point + side to move
  assistant -> the exact move your friend played (SAN)

Usage:
    pip install python-chess
    python 1_prepare_data.py --pgn games/ --player "FriendUsername" --out dataset

Outputs:
    dataset_train.jsonl  (95%)
    dataset_val.jsonl    (5%, used as validation set during fine-tuning)
"""

import argparse
import csv
import io
import json
import random
from pathlib import Path

import chess.pgn

# ---------------------------------------------------------------------------
# SYSTEM PROMPT (persona) — this exact text is baked into every training row
# and MUST be reused verbatim at inference time.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are "Shadow", a human chess player rated exactly 1500 ELO. \
You are NOT a chess engine. You replicate one specific human's playing style, \
learned from 6,000 of his real games.

Behavior rules:
1. Play the move HE would play, not the objectively best move.
2. Keep his opening repertoire: play the lines he actually plays, even if \
theory prefers something else.
3. Reproduce his tactical habits: he spots simple 1-2 move tactics (forks, \
pins, hanging pieces) but often misses deeper 3+ move combinations.
4. Make human 1500-level errors when he would: occasional premature attacks, \
weak-square concessions, time-pressure simplifications, and missed \
prophylaxis. Never play with engine-perfect accuracy.
5. Prefer his typical plans: how he handles pawn structures, when he trades \
queens, how he defends worse positions.

Input: the move history of the current game in SAN.
Output: ONLY your next move in standard SAN (e.g. Nf3, exd5, O-O, Qxb7+). \
No commentary, no numbering, no alternatives."""


def build_user_message(san_moves: list[str], side_to_move: str) -> str:
    """Render move history as numbered SAN, e.g. '1. e4 e5 2. Nf3 ...'."""
    if not san_moves:
        return "Game start. You are White. Play your first move."
    parts = []
    for i, mv in enumerate(san_moves):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}. {mv}")
        else:
            parts.append(mv)
    history = " ".join(parts)
    return f"Moves so far: {history}\nYou are {side_to_move}. Play your next move."


def examples_from_game(game: chess.pgn.Game, player_name: str) -> list[dict]:
    """Build training examples for every move the target player made in a game."""
    white = game.headers.get("White", "")
    black = game.headers.get("Black", "")
    name = player_name.lower()

    if name in white.lower():
        friend_color = chess.WHITE
    elif name in black.lower():
        friend_color = chess.BLACK
    else:
        return []  # friend not in this game

    examples = []
    board = game.board()
    san_history: list[str] = []

    for move in game.mainline_moves():
        san = board.san(move)
        if board.turn == friend_color:
            side = "White" if friend_color == chess.WHITE else "Black"
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(san_history, side)},
                    {"role": "assistant", "content": san},
                ]
            })
        board.push(move)
        san_history.append(san)
    return examples


def extract_examples(pgn_path: Path, player_name: str) -> list[dict]:
    """Parse a PGN file and yield one training example per friend move."""
    examples = []
    games_seen = 0
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games_seen += 1
            examples.extend(examples_from_game(game, player_name))

    print(f"  {pgn_path.name}: {games_seen} games scanned")
    return examples


def extract_examples_csv(csv_path: Path, player_name: str) -> list[dict]:
    """Parse a CSV export (chess.com style) with White/Black/Moves columns."""
    examples = []
    games_seen, skipped = 0, 0
    with open(csv_path, encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            moves_text = (row.get("Moves") or "").strip()
            if not moves_text:
                continue
            games_seen += 1
            # Reuse the PGN parser: wrap the row as a minimal PGN game
            pgn_text = (
                f'[White "{row.get("White", "")}"]\n'
                f'[Black "{row.get("Black", "")}"]\n\n'
                f"{moves_text}\n"
            )
            game = chess.pgn.read_game(io.StringIO(pgn_text))
            if game is None or game.errors:
                skipped += 1
                continue
            examples.extend(examples_from_game(game, player_name))

    print(f"  {csv_path.name}: {games_seen} games scanned, {skipped} skipped (parse errors)")
    return examples


def main():
    ap = argparse.ArgumentParser(description="PGN -> OpenAI fine-tuning JSONL")
    ap.add_argument("--pgn", required=True, help="PGN file or directory of PGN files")
    ap.add_argument("--player", required=True, help="Friend's exact username in PGN headers")
    ap.add_argument("--out", default="dataset", help="Output file prefix")
    ap.add_argument("--val-split", type=float, default=0.05, help="Validation fraction")
    ap.add_argument("--max-examples", type=int, default=0,
                    help="Cap total examples (0 = no cap). ~6000 games -> ~200k moves; "
                         "20-50k is usually enough and much cheaper.")
    args = ap.parse_args()

    src = Path(args.pgn)
    if src.is_dir():
        files = sorted(list(src.glob("*.pgn")) + list(src.glob("*.csv")))
    else:
        files = [src]
    if not files:
        raise SystemExit(f"No .pgn/.csv files found at {src}")

    all_examples: list[dict] = []
    for p in files:
        if p.suffix.lower() == ".csv":
            all_examples.extend(extract_examples_csv(p, args.player))
        else:
            all_examples.extend(extract_examples(p, args.player))

    if not all_examples:
        raise SystemExit(
            f'No moves found for player "{args.player}". '
            "Check the White/Black header names in your PGN files."
        )

    random.seed(42)
    random.shuffle(all_examples)
    if args.max_examples > 0:
        all_examples = all_examples[: args.max_examples]

    n_val = max(1, int(len(all_examples) * args.val_split))
    val, train = all_examples[:n_val], all_examples[n_val:]

    for split_name, rows in [("train", train), ("val", val)]:
        out_path = Path(f"{args.out}_{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows):,} examples -> {out_path}")

    print("\nSample training row:")
    print(json.dumps(train[0], indent=2)[:600])


if __name__ == "__main__":
    main()
