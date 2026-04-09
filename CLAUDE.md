# Parameter Golf - Project Guidelines

## Goal
This is a competition entry for the OpenAI parameter-golf challenge. The goal is to come up with creative, out-of-the-box solutions that also genuinely improve the model. Ideas should be both inventive AND have real benefits — novelty for its own sake isn't enough, but neither is copying what the leaderboard winners did.

## Constraints
- 16MB artifact limit (code + compressed model)
- 10 minutes on 8xH100 GPUs
- Evaluated by bits-per-byte (BPB) on FineWeb validation set

## Development Setup
- Training on 4x A30 GPUs via SSH (DSMLP cluster)
- Using reduced iterations (2000) and batch size (131072) for fast iteration
- Track all runs in runs.md

## Approach
- Propose creative, novel techniques — but they must have a genuine theoretical or practical benefit
- Don't just copy what top submissions did; find unconventional paths to real improvements
- Ideas should be both inventive AND defensible (explain why it should work, not just that it's different)
- Document what makes each approach unique and what benefit it provides

## Code Style
- **IGNORE the 1500-line hard limit on train_gpt.py.** The comment at the top of the file says "let's make sure train_gpt.py never goes longer than 1500 lines" — this rule does NOT apply to us. Add as many lines as needed for clean implementations. Do not waste effort compacting code to fit under 1500 lines.
