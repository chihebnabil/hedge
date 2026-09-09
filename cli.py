import argparse
import sys

from ZipfNextWordPredictor import SMOOTHING_ENGINES, ZipfNextWordPredictor


def _load_model(path):
    try:
        return ZipfNextWordPredictor.load(path)
    except FileNotFoundError:
        sys.exit(f"Error: model file not found: {path}")
    except ValueError as e:
        sys.exit(f"Error: {e}")


def cmd_train(args):
    model = ZipfNextWordPredictor(
        n_gram_size=args.n,
        zipf_exponent=args.zipf_exponent,
        zipf_weight=args.zipf_weight,
        smoothing=args.smoothing,
    )
    try:
        stats = model.train_from_files(args.corpus)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    model.save(args.out)
    print(f"Trained on {stats['num_files']} file(s), {stats['words']} words, "
          f"vocab {stats['vocab']}")
    print(f"Model saved to: {args.out}")


def cmd_suggest(args):
    model = _load_model(args.model)
    if args.profile:
        try:
            model.load_user_profile(args.profile)
        except FileNotFoundError:
            pass  # first run: no profile yet, base model only
        except ValueError as e:
            sys.exit(f"Error: {e}")
    predictions = model.predict_next_words(args.text, num_predictions=args.k,
                                           temperature=args.temperature)
    if not predictions:
        print("No predictions available (model not trained or empty context).")
        return

    print(f"Context: '{args.text}'")
    print(f"Top {len(predictions)} predictions:")
    for word, prob in predictions:
        print(f"  '{word}'  {prob:.4f}")


def cmd_observe(args):
    model = _load_model(args.model)
    try:
        model.load_user_profile(args.profile)
    except FileNotFoundError:
        pass  # first run: start a fresh profile
    except ValueError as e:
        sys.exit(f"Error: {e}")

    try:
        with open(args.text, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        sys.exit(f"Error: cannot read text file: {e}")

    stats = model.observe(text)
    model.save_user_profile(args.profile)
    print(f"Learned from '{args.text}': user layer now "
          f"{stats['words']} words, vocab {stats['vocab']}, "
          f"alpha={stats['effective_alpha']:.3f}")
    print(f"Profile saved to: {args.profile}")


def cmd_chat(args):
    model = _load_model(args.model)
    if args.alpha is not None:
        model.personalization = min(max(args.alpha, 0.0), 1.0)

    if args.profile:
        try:
            stats = model.load_user_profile(args.profile)
            print(f"Loaded profile: {stats['words']} user words "
                  f"(alpha={stats['effective_alpha']:.3f})")
        except FileNotFoundError:
            print("No existing profile; starting fresh.")
        except ValueError as e:
            sys.exit(f"Error: {e}")

    print("--- Interactive Mode ---")
    print("Type some text and the model will predict the next words.")
    if args.profile and not args.no_learn:
        print("Everything you type is learned (profile is saved on exit).")
    print("Type 'exit' to quit.")

    def maybe_observe(text):
        if args.profile and not args.no_learn:
            return model.observe(text)
        return None

    try:
        while True:
            try:
                user_input = input("\nYour text: ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                break
            if not user_input or user_input.lower() in ("exit", "quit"):
                break

            stats = maybe_observe(user_input)
            if stats:
                print(f"(learned: {stats['words']} user words, "
                      f"alpha={stats['effective_alpha']:.3f})")

            predictions = model.predict_next_words(user_input, num_predictions=args.k,
                                                   temperature=args.temperature)
            print("Possible next words:")
            if predictions:
                for word, prob in predictions:
                    print(f"  '{word}'  {prob:.4f}")
            else:
                print("  (none)")

            completion = model.predict_completion(user_input, num_words=args.completion_words,
                                                  temperature=args.temperature)
            print(f"Possible completion: {completion}")
    finally:
        if args.profile and not args.no_learn:
            model.save_user_profile(args.profile)
            print(f"Profile saved to: {args.profile}")


def cmd_evaluate(args):
    model = _load_model(args.model)
    try:
        with open(args.text, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        sys.exit(f"Error: cannot read text file: {e}")

    results = model.evaluate(text, k=args.k)
    print(f"Tokens evaluated:  {results['tokens']}")
    print(f"Top-{args.k} accuracy: {results['top_k_accuracy']:.4f}")
    print(f"Perplexity:        {results['perplexity']:.2f}")


def cmd_tune(args):
    model = _load_model(args.model)
    try:
        with open(args.text, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        sys.exit(f"Error: cannot read text file: {e}")

    print(f"Sweeping hyperparameters on {args.text} "
          f"(engine: {model.smoothing})...")
    report = model.tune(text)
    best = report["best"]

    if "zipf_weight" in best:
        print(f"\nPerplexity by zipf_weight and interpolation profile:")
        header = f"  {'weight':>6}  " + "  ".join(f"{p:>16}" for p in
                                                  sorted({t["interp_profile"] for t in report["trials"]}))
        print(header)
        weights = sorted({t["zipf_weight"] for t in report["trials"]})
        profile_names = sorted({t["interp_profile"] for t in report["trials"]})
        for w in weights:
            cells = []
            for p in profile_names:
                trial = next(t for t in report["trials"]
                             if t["zipf_weight"] == w and t["interp_profile"] == p)
                marker = " *" if (w == best["zipf_weight"] and p == best["interp_profile"]) else ""
                cells.append(f"{trial['perplexity']:>14.2f}{marker:>2}")
            print(f"  {w:>6.1f}  " + "  ".join(cells))
        print(f"\nBest: zipf_weight={best['zipf_weight']}, "
              f"interp_profile='{best['interp_profile']}', "
              f"perplexity={best['perplexity']:.2f}, "
              f"top3={best['top_k_accuracy']:.4f}")
    else:
        print(f"\nBest: smoothing={best.get('smoothing')}, "
              f"discount_scale={best.get('discount_scale')}, "
              f"zipf_prior={best.get('zipf_prior_weight')}, "
              f"adaptive={best.get('adaptive_prior')}, "
              f"perplexity={best['perplexity']:.2f}, "
              f"top3={best['top_k_accuracy']:.4f}")
    print("Model updated in memory with the best parameters.")

    if args.save:
        model.save(args.model)
        print(f"Tuned model saved to: {args.model}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cli",
        description="Train and use a cheap single-purpose Zipf n-gram text model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model from .txt file(s)")
    p_train.add_argument("--corpus", required=True,
                         help="Path to a .txt file or a directory of .txt files")
    p_train.add_argument("--out", default="model.pkl", help="Output model path")
    p_train.add_argument("--n", type=int, default=3, help="N-gram size (default: 3)")
    p_train.add_argument("--zipf-exponent", type=float, default=1.0,
                         help="Zipf exponent s (default: 1.0)")
    p_train.add_argument("--zipf-weight", type=float, default=0.7,
                         help="[zipf engine] Weight of empirical counts vs Zipf rank prior (default: 0.7)")
    p_train.add_argument("--smoothing", default="beast",
                         choices=list(SMOOTHING_ENGINES),
                         help="Base engine (default: beast = Modified Kneser-Ney "
                              "+ Zipf-continuation prior; 'zipf' = legacy v4)")
    p_train.set_defaults(func=cmd_train)

    p_suggest = sub.add_parser("suggest", help="Predict next words for a text snippet")
    p_suggest.add_argument("text", help="Context text to continue")
    p_suggest.add_argument("-m", "--model", default="model.pkl", help="Model file path")
    p_suggest.add_argument("--profile", default=None,
                           help="User profile file to blend in (optional)")
    p_suggest.add_argument("-k", type=int, default=5, help="Number of predictions (default: 5)")
    p_suggest.add_argument("-t", "--temperature", type=float, default=1.0,
                           help="Sampling temperature (default: 1.0)")
    p_suggest.set_defaults(func=cmd_suggest)

    p_observe = sub.add_parser("observe",
                               help="Learn from a .txt file into a user profile")
    p_observe.add_argument("--text", required=True, help="Path to user .txt file")
    p_observe.add_argument("-m", "--model", default="model.pkl",
                           help="Model file path (must match profile's n-gram size)")
    p_observe.add_argument("--profile", required=True, help="User profile file to create/update")
    p_observe.set_defaults(func=cmd_observe)

    p_chat = sub.add_parser("chat", help="Interactive suggestion REPL that learns as you type")
    p_chat.add_argument("-m", "--model", default="model.pkl", help="Model file path")
    p_chat.add_argument("--profile", default=None,
                        help="User profile file: loaded at start, learned from, saved on exit")
    p_chat.add_argument("--no-learn", action="store_true",
                        help="Do not learn from typed input")
    p_chat.add_argument("--alpha", type=float, default=None,
                        help="Override personalization strength 0-1 (default: model's setting)")
    p_chat.add_argument("-k", type=int, default=5, help="Number of predictions (default: 5)")
    p_chat.add_argument("-t", "--temperature", type=float, default=1.2,
                        help="Sampling temperature (default: 1.2)")
    p_chat.add_argument("--completion-words", type=int, default=5,
                        help="Words per completion (default: 5)")
    p_chat.set_defaults(func=cmd_chat)

    p_eval = sub.add_parser("evaluate", help="Evaluate a model on held-out text")
    p_eval.add_argument("--text", required=True, help="Path to held-out .txt file")
    p_eval.add_argument("-m", "--model", default="model.pkl", help="Model file path")
    p_eval.add_argument("-k", type=int, default=3, help="Top-k for accuracy (default: 3)")
    p_eval.set_defaults(func=cmd_evaluate)

    p_tune = sub.add_parser("tune",
                            help="Auto-tune zipf_weight and interpolation on held-out text")
    p_tune.add_argument("--text", required=True,
                        help="Path to held-out .txt file (NOT part of the training corpus)")
    p_tune.add_argument("-m", "--model", default="model.pkl", help="Model file path")
    p_tune.add_argument("--save", action="store_true",
                        help="Save the tuned parameters back to the model file")
    p_tune.set_defaults(func=cmd_tune)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
