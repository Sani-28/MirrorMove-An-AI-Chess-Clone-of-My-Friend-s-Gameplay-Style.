"""
PLAY vs your chess style clone
==============================
Loads the fine-tuned LoRA adapter and lets you play a full game in the
terminal. You are White; the model plays Black in the cloned style.

Requires a CUDA GPU (works great in a Kaggle/Colab notebook too).

Usage:
    pip install -r requirements.txt
    python play.py --model path/to/chess_lora
"""

import argparse
import random

import chess
from unsloth import FastLanguageModel

# MUST match the system prompt used during training (prepare_data.py)
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


def build_user_message(san_moves: list[str], side: str) -> str:
    if not san_moves:
        return "Game start. You are White. Play your first move."
    parts = []
    for i, mv in enumerate(san_moves):
        parts.append(f"{i // 2 + 1}. {mv}" if i % 2 == 0 else mv)
    return f"Moves so far: {' '.join(parts)}\nYou are {side}. Play your next move."


def predict(model, tokenizer, san_moves: list[str], side: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(san_moves, side)},
    ]
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(input_ids=inputs, max_new_tokens=8,
                         temperature=0.3, do_sample=True)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description="Play against your chess clone")
    ap.add_argument("--model", required=True, help="Path to trained chess_lora folder")
    args = ap.parse_args()

    print("Loading model (this takes a few minutes)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=1024, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    print("Model ready! You are White. Enter moves in SAN (e4, Nf3, O-O). 'quit' to exit.\n")

    board, history = chess.Board(), []

    while not board.is_game_over():
        print(board, "\n")
        if board.turn == chess.WHITE:
            san = input("Your move: ").strip()
            if san.lower() == "quit":
                break
            try:
                move = board.parse_san(san)
            except ValueError:
                print("Illegal move, try again.\n")
                continue
        else:
            move = None
            for _ in range(3):  # retry on illegal model output
                s = predict(model, tokenizer, history, "Black").split()[0].strip(".!?")
                try:
                    move = board.parse_san(s)
                    break
                except ValueError:
                    continue
            if move is None:  # fallback: random legal move
                move = random.choice(list(board.legal_moves))
            print(f"Clone plays: {board.san(move)}\n")

        history.append(board.san(move))
        board.push(move)

    print(board)
    print("Result:", board.result())


if __name__ == "__main__":
    main()
