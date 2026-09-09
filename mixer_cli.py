"""
mixer_cli.py — command-line interface for the expert-mixer LM.

Commands:
  suggest "some context" -k 5     top-k next-word suggestions
  complete "some context" -n 6    greedy continuation
  observe FILE_OR_TEXT            feed user text to the personalization layer
  score FILE                      perplexity of a text file under the mixer
  demo                            scripted end-to-end tour (real prompts)
  chat                            REPL: type a line, get suggestions;
                                  '>>' prefix completes the line instead;
                                  '!observe text' personalizes; empty quits

Online mode: weights adapt live (eta=0.1) — the deployment story is
"model keeps learning from its user"; benchmark mode (mixer_lm.py) is the
frozen-weights protocol.

Examples:
  python mixer_cli.py demo
  python mixer_cli.py suggest "once upon a" -k 5
  python mixer_cli.py observe "the princess decided to write python code"
  python mixer_cli.py suggest "the princess decided to" -k 5
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mixer_lm import build, warm_up  # noqa: E402


def get_mixer(with_user=True, tune=False):
    streams, mixer = build(with_user=with_user)
    if tune:
        warm_up(mixer, streams["valid"][:2500], verbose=False)
        mixer.eta = 0.1          # resume gentle online adaptation
    else:
        mixer.eta = 0.1
    return mixer, streams


def fmt(cands):
    return ", ".join(f"{w} ({p:.3f})" for w, p in cands)


def show(mixer, line, k=5):
    words = [w for w in line.lower().split() if w]
    cands = mixer.topk(words, k)
    print(f"  context : {' '.join(words[-6:])}")
    print(f"  suggest : {fmt(cands)}")


def cmd_suggest(args, mixer, _streams):
    show(mixer, args.text, args.k)


def cmd_complete(args, mixer, _streams):
    words = [w for w in args.text.lower().split() if w]
    out = mixer.complete(words, args.n)
    print(f"  {' '.join(words)} {' '.join(out)}")


def cmd_observe(args, mixer, _streams):
    if os.path.isfile(args.text):
        with open(args.text, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = args.text
    mixer.observe(text)
    user = next(e for e in mixer.experts if e.name == "user")
    print(f"  observed {len(user.win)} user words "
          f"({len(user.counts)} unique)")


def cmd_score(args, mixer, _streams):
    with open(args.file, "r", encoding="utf-8", errors="replace") as f:
        sents = [line.lower().split() for line in f if line.strip()]
    r = mixer.run(sents)
    print(f"  PP={r['pp_mix']:.2f}  tokens={r['n']}")


def cmd_chat(_args, mixer, _streams):
    print("Mixer chat — type text, get suggestions. '>>text' completes. "
          "'!observe text' personalizes. Empty line quits.")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        if line.startswith("!observe "):
            mixer.observe(line[9:])
            user = next(e for e in mixer.experts if e.name == "user")
            print(f"  [observed; {len(user.win)} user words]")
            continue
        if line.startswith(">>"):
            words = [w for w in line[2:].lower().split() if w]
            out = mixer.complete(words, 8)
            print(f"  {' '.join(words)} {' '.join(out)}")
            for w in out:
                for e in mixer.experts:
                    if e.name == "cache":
                        e.update(mixer.w2i.get(w, mixer.unk))
            continue
        show(mixer, line)
        for e in mixer.experts:          # the typed words become history
            if e.name == "cache":
                for w in line.lower().split():
                    e.update(mixer.w2i.get(w, mixer.unk))


def cmd_demo(_args, mixer, streams):
    print("=" * 68)
    print("MIXER LM — END-TO-END DEMO (WikiText-2 trained, 6 experts, "
          "online)")
    print("=" * 68)

    prompts = [
        "once upon a",
        "the united states of",
        "as a result of the",
        "it was the first time",
    ]
    print("\n[1] Real prompts, top-5 suggestions (fresh model, no tuning):")
    for p in prompts:
        show(mixer, p)

    print("\n[2] Greedy completions:")
    for p in ("the story begins", "he did not know what"):
        words = p.split()
        print(f"  {p} -> {p} {' '.join(mixer.complete(words, 6))}")

    print("\n[3] Quality probe: perplexity of 500 held-out test sentences")
    r = mixer.run(streams["test"][:500])
    print(f"  PP={r['pp_mix']:.2f} over {r['n']:,} tokens "
          f"(online, from scratch — no tuning)")

    print("\n[4] Personalization: model learns from a short user text")
    print("  before:", fmt(mixer.topk("walked in the".split(), 5)))
    print("  alphas :", mixer.alphas_for("walked in the".split()))
    mixer.observe("the princess walked in the garden every morning. "
                  "the princess loved the quiet garden near the castle. "
                  "every morning the princess sang in the garden, and the "
                  "people of the castle listened to the princess. "
                  "the garden was the most beautiful place in the kingdom, "
                  "and the princess loved it more every morning.")
    print("  -- after observing ~55 user words --")
    print("  after :", fmt(mixer.topk("walked in the".split(), 5)))
    print("  alphas :", mixer.alphas_for("walked in the".split()))
    print("  completion:", " ".join(
        mixer.complete("every morning the princess".split(), 4)))

    print("\n[5] Domain shift probe: suggestions after reading a cooking "
          "paragraph")
    mixer.observe("stir the sauce over low heat. add the onions to the "
                  "sauce and cook until soft. season the sauce with salt "
                  "and pepper. pour the sauce over the vegetables and "
                  "serve the dish with bread. the sauce thickens as it "
                  "cooks, so stir the sauce often and keep the heat low.")
    print("  alphas :", mixer.alphas_for("stir the".split()))
    show(mixer, "stir the")
    show(mixer, "season the")
    print("\ndone.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("suggest")
    s.add_argument("text")
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(fn=cmd_suggest)

    s = sub.add_parser("complete")
    s.add_argument("text")
    s.add_argument("-n", type=int, default=6)
    s.set_defaults(fn=cmd_complete)

    s = sub.add_parser("observe")
    s.add_argument("text")
    s.set_defaults(fn=cmd_observe)

    s = sub.add_parser("score")
    s.add_argument("file")
    s.set_defaults(fn=cmd_score)

    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    sub.add_parser("chat").set_defaults(fn=cmd_chat)

    args = p.parse_args()
    with_user = args.cmd in ("suggest", "complete", "observe", "chat", "demo")
    mixer, streams = get_mixer(with_user=with_user)
    args.fn(args, mixer, streams)


if __name__ == "__main__":
    main()
