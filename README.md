# Chess Style Clone — Fine-tuning an LLM to Play Like a Specific Human

Train a large language model to mimic the exact chess playing style of one specific player, using only their game history. This project takes ~6,000 real games of a 1500-ELO chess.com player and fine-tunes **Qwen2.5-7B-Instruct** with LoRA so the model plays *his* openings, *his* plans — and *his* mistakes.

Unlike Stockfish (which plays perfect moves) this model answers a different question: **"In this position, what would HE play?"**

> Built entirely on free tools: Kaggle's free GPU (T4), Unsloth, and open-source models. Total cost: $0.

---

## How It Works

```
chess.com games (CSV/PGN)
        │
        ▼
prepare_data.py ──► JSONL chat dataset
        │             system: persona prompt ("you are a 1500 ELO human...")
        │             user:   move history in SAN ("1. e4 e5 2. Nf3 ...")
        │             assistant: the exact move he played
        ▼
kaggle_chess_finetune.ipynb ──► LoRA fine-tune on Kaggle T4 GPU
        │                        Qwen2.5-7B-Instruct, 4-bit, ~0.5% params trained
        ▼
play.py ──► play a full game against the clone in your terminal
```

Every move the target player made becomes one training example. The model learns to predict his next move given the game so far.

## Results

| Metric | Value |
|---|---|
| Training data | 6,081 games → 19,000 positions |
| Training | 400 steps (~1/3 epoch), ~2.7 hrs on Kaggle T4 |
| Loss | 1.36 → 0.86 |
| Exact-move match (100 val positions) | **11%** |
| Opening repertoire | ✅ Correctly reproduces his lines (e4, Nf3, Bc4 Italian setup) |

The clone plays the target player's opening repertoire convincingly, but middlegame accuracy is limited (see Limitations).

## How to Run

### 1. Prepare the dataset

```bash
pip install python-chess
python prepare_data.py --pgn your_games.csv --player YourUsername --out dataset --max-examples 20000
```

Accepts PGN files or chess.com CSV exports (with `White`, `Black`, `Moves` columns). Produces `dataset_train.jsonl` and `dataset_val.jsonl`.

### 2. Fine-tune on Kaggle (free)

1. Upload both JSONL files as a Kaggle Dataset
2. Import `kaggle_chess_finetune.ipynb` into a Kaggle Notebook
3. Settings: **Accelerator = GPU T4 x2**, **Internet = ON**, Add Input → your dataset
4. **Save Version → Save & Run All (Commit)** — trains in the background (~3 hrs)
5. Download the `chess_lora` adapter from the notebook Output

### 3. Play against the clone

```bash
pip install -r requirements.txt
python play.py --model path/to/chess_lora
```

Requires a CUDA GPU — running it inside a Kaggle notebook works too.

## Limitations (honest ones)

- **11% exact-move match is low.** Board-state approaches like [Maia Chess](https://arxiv.org/abs/2006.01855) reach ~50% by feeding the network the actual board as an 8×8 tensor. My text-based approach makes the model reconstruct the board from the move list, which gets unreliable in long games.
- **Undertrained.** Only 1/3 of one epoch (Kaggle session-time constraints). A full epoch should reach ~20% match.
- **Occasional illegal moves.** The model sometimes hallucinates illegal moves; `play.py` validates with python-chess and retries.
- **Middlegame > move 25 degrades** as the context gets longer and the model loses track of the position.

## What I Learned / Future Work

- LLM-as-chess-player is a fun experiment but a **policy network on board tensors (Maia-style) is the right architecture** for this problem — planned as v2
- Feed FEN (current board state) instead of full move history
- Full-epoch training via Kaggle Commit mode
- Stockfish blunder filter to veto catastrophic moves
- Lichess bot integration so friends can challenge the clone online

## Tech Stack

Qwen2.5-7B-Instruct · Unsloth (LoRA, 4-bit QLoRA) · TRL/PEFT · python-chess · Kaggle T4 GPU

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for 2x faster free fine-tuning
- [Maia Chess](https://maiachess.com/) for the research inspiration on human-like chess AI
- My friend, whose 6,000 games (and 1500-ELO blunders) made this possible 😄
